import torch.nn.functional as F
import math
import torch
from nltk.tokenize import word_tokenize
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import copy
import numpy as np
import os
import re
import sacrebleu
import random
import time
import jieba
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]

    def __len__(self):
        return len(self.src)

class Tokenizer():
    def __init__(self, en_path, ch_path, count_min=5):
        self.en_path = en_path
        self.ch_path = ch_path
        self.__count_min = count_min

        self.en_data = self.__read_ori_data(en_path)
        self.ch_data = self.__read_ori_data(ch_path)

        self.index_2_word = ['unK', '<pad>', '<bos>', '<eos>']
        self.word_2_index = {'unK': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}

        self.en_set = set()
        self.en_count = {}
        self.ch_set = set()  # 新增中文集合
        self.ch_count = {}  # 新增中文计数器

        self.__count_word()
        self.mx_length = 40
        self.data_ = []
        self.__filter_data()
        random.shuffle(self.data_)
        self.test = self.data_[-1000:]
        self.data_ = self.data_[:-1000]

    def __read_ori_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __count_word(self):
        le = len(self.en_data)
        p = 0

        # 统计英文词汇表
        for data in self.en_data:
            if p % 1000 == 0:
                print('英文处理进度:', p / le)
            sentence = word_tokenize(data)
            for tk in sentence:
                if tk in self.en_set:
                    self.en_count[tk] += 1
                else:
                    self.en_set.add(tk)
                    self.en_count[tk] = 1
            p += 1
        for word, count in self.en_count.items():
            if count >= self.__count_min:
                self.word_2_index[word] = len(self.index_2_word)
                self.index_2_word.append(word)
            else:
                self.word_2_index[word] = 0

        p = 0
        # 统计中文词汇表
        for data in self.ch_data:
            if p % 1000 == 0:
                print('中文处理进度:', p / le)
            sentence = list(jieba.cut(data))
            for tk in sentence:
                if tk in self.ch_set:  # 使用中文集合
                    self.ch_count[tk] += 1
                else:
                    self.ch_set.add(tk)  # 使用中文集合
                    self.ch_count[tk] = 1
            p += 1
        for word, count in self.ch_count.items():  # 使用中文计数器
            if count >= self.__count_min:
                self.word_2_index[word] = len(self.index_2_word)
                self.index_2_word.append(word)
            else:
                self.word_2_index[word] = 0

    def __filter_data(self):
        length = len(self.en_data)
        for i in range(length):
            self.data_.append([self.en_data[i], self.ch_data[i], 0])
            self.data_.append([self.ch_data[i], self.en_data[i], 1])

    def en_cut(self, data):
        data = word_tokenize(data)
        if len(data) > self.mx_length:
            return 0, []
        en_tokens = [self.word_2_index.get(tk, 0) for tk in data]
        return 1, en_tokens

    def ch_cut(self, data):
        data = list(jieba.cut(data))
        if len(data) > self.mx_length:
            return 0, []
        ch_tokens = [self.word_2_index.get(tk, 0) for tk in data]
        return 1, ch_tokens

    def encode_all(self, data):
        src, tgt = [], []
        valid_data = []
        for item in data:
            src_sent, tgt_sent, label = item
            if label == 0:
                lab1, src_tokens = self.en_cut(src_sent)
                lab2, tgt_tokens = self.ch_cut(tgt_sent)
            else:
                lab1, tgt_tokens = self.en_cut(tgt_sent)
                lab2, src_tokens = self.ch_cut(src_sent)
            if lab1 and lab2:
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                valid_data.append(item)
        return valid_data, src, tgt

    def encode(self, src, label):
        if label == 0:
            tokens = word_tokenize(src)
        else:
            tokens = list(jieba.cut(src))
        return [self.word_2_index.get(tk, 0) for tk in tokens if len(tokens) <= self.mx_length] or [0]

    def decode(self, data):
        return self.index_2_word[data]

    def __get_datasets(self, data):
        valid_data, src, tgt = self.encode_all(data)
        return TranslationDataset(src, tgt) if src else None

    def another_process(self, batch_datas):
        en_index, ch_index = [], []
        en_len, ch_len = [], []

        for en, ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        max_en_len = max(en_len) if en_len else 0
        max_ch_len = max(ch_len) if ch_len else 0
        max_len = max(max_en_len, max_ch_len + 2)

        en_index = [i + [self.word_2_index['<pad>']] * (max_len - len(i)) for i in en_index]
        ch_index = [[self.word_2_index['<bos>']] + i + [self.word_2_index['<eos>']] +
                    [self.word_2_index['<pad>']] * (max_len - len(i) - 1) for i in ch_index]

        return torch.tensor(en_index), torch.tensor(ch_index)

    def get_dataloader(self, data, batch_size=40):
        dataset = self.__get_datasets(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.another_process) if dataset else None

    def get_vocab_size(self):
        return len(self.index_2_word)

    def get_dataset_size(self):
        return len(self.en_data)

class Batch:
    def __init__(self, src, trg=None, tokenizer=None, device='cuda'):
        src = src.to(device).long()
        trg = trg.to(device).long() if trg is not None else None
        self.src = src
        self.__pad = tokenizer.word_2_index['<pad>']
        self.src_mask = (src != self.__pad).unsqueeze(-2)
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.ntokens = 0

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            self.ntokens = (self.trg_y != self.__pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0.0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (-math.log(1e4) / d_model)).unsqueeze(0)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x = self.norm(x)
        x = sublayer(x)
        x = self.dropout(x)
        return x + x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, tokenizer, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        self.src_embed = Embedding(d_model, tokenizer.get_vocab_size())
        self.tgt_embed = Embedding(d_model, tokenizer.get_vocab_size())
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        self.generator = Generator(d_model, tokenizer.get_vocab_size())
        self.device = device

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        for encoder_layer in self.encoder:
            src = encoder_layer(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, tgt, src_mask, tgt_mask)

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

smooth = SmoothingFunction().method1
def compute_bleu4(tokenizer, test_indices, model, device):
    """计算 BLEU-4 分数"""
    model.eval()
    bleu_scores = []
    
    with torch.no_grad():
        for i in test_indices:
            # 获取源句子和参考译文
            src_sent = tokenizer.test[i][0]  # 源语言句子
            ref_sent = tokenizer.test[i][1]  # 参考译文（字符串）
            
            # 准备输入
            src = torch.tensor([tokenizer.word_2_index.get(w, tokenizer.word_2_index['<unk>']) for w in src_sent], device=device).unsqueeze(0)
            src_mask = (src != tokenizer.word_2_index['<pad>']).unsqueeze(-2)
            
            # 模型推理
            memory = model.encode(src, src_mask)
            ys = torch.ones(1, 1).fill_(tokenizer.word_2_index['<sos>']).type_as(src.data)
            
            # 贪婪解码
            for _ in range(100):
                out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                
                if next_word == tokenizer.word_2_index['<eos>']:
                    break
            
            # 将预测的 token ID 转换为文本
            hyp_tokens = ys[0, 1:].tolist()  # 去掉 <sos>
            hyp_tokens = [tokenizer.index_2_word[idx] for idx in hyp_tokens if idx != tokenizer.word_2_index['<eos>']]
            hyp_text = ' '.join(hyp_tokens)  # 转换为字符串
            
            # 计算 BLEU 分数
            bleu = sacrebleu.sentence_bleu(hyp_text, [ref_sent]).score / 100  # 确保 hyp_text 是字符串
            bleu_scores.append(bleu)
    
    # 计算平均 BLEU 分数
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
    for i in range(max_len - 1):
        out = model.decode(memory, ys, src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).long().fill_(next_word)], dim=1)
        if next_word == 3:  # EOS
            break
    return ys.squeeze(0)

def predict(data, model, tokenizer, device='cuda'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for src_sent, _, label in data:
            src_tokens = tokenizer.encode(src_sent, label)
            src = torch.tensor([src_tokens], device=device).long()
            src_mask = (src != tokenizer.word_2_index['<pad>']).unsqueeze(-2)
            output = greedy_decode(model, src, src_mask, max_len=100, start_symbol=2)
            pred = []
            for idx in output.tolist()[1:]:  # 跳过BOS
                if idx == 3:  # EOS
                    break
                pred.append(tokenizer.index_2_word[idx])
            if label == 0:
                predictions.append(''.join(pred))
            else:
                predictions.append(TreebankWordDetokenizer().detokenize(pred))
    return predictions
# 新增分布式相关导入
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def train(en_path, ch_path, model_save_path='./models/'):
    # 1. 初始化分布式环境
    # 从环境变量获取分布式信息（由启动脚本传递）
    rank = int(os.environ['RANK'])  # 进程编号
    local_rank = int(os.environ['LOCAL_RANK'])  # 本地 GPU 编号
    world_size = int(os.environ['WORLD_SIZE'])  # 总进程数

    # 初始化 NCCL 后端（支持 GPU 通信）
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # 从环境变量获取 master 地址和端口
        rank=rank,
        world_size=world_size
    )

    # 设置当前进程使用的 GPU
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # 2. 加载 Tokenizer（仅主进程打印日志）
    if rank == 0:
        print("Loading tokenizer...")
    tokenizer = Tokenizer(en_path, ch_path, count_min=3)

    # 3. 初始化模型并同步参数到所有进程
    model = Transformer(tokenizer, device=device)
    # 初始化参数（仅主进程初始化，其他进程同步）
    if rank == 0:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    # 同步所有进程的模型参数
    for p in model.parameters():
        dist.broadcast(p.data, src=0)  # 主进程（rank=0）广播参数到其他进程
    model.to(device)

    # 4. 封装为 DistributedDataParallel 模型
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],  # 指定当前进程使用的 GPU
        output_device=local_rank,
        find_unused_parameters=False  # 根据模型结构调整（若有未使用参数设为 True）
    )

    # 5. 数据加载（使用 DistributedSampler 分片）
    # 训练数据加载器（每个进程处理 1/world_size 的数据）
    train_dataset = tokenizer._get_datasets(tokenizer.data_)
    if not train_dataset:
        if rank == 0:
            print("No valid training data")
        return

    # 分布式采样器（自动分片数据）
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # 每个 epoch 打乱数据分片
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 单卡 batch_size（总 batch_size = 32 * world_size）
        sampler=train_sampler,  # 使用分布式采样器
        collate_fn=tokenizer.another_process,
        pin_memory=True  # 加速数据从 CPU 到 GPU 的拷贝
    )

    # 6. 优化器与损失函数
    criteria = LabelSmoothing(tokenizer.get_vocab_size(), tokenizer.word_2_index['<pad>'], smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    opt = NoamOpt(256, 1, 2000, optimizer)
    loss_compute = SimpleLossCompute(model.generator, criteria, opt)

    # 7. 训练循环（仅主进程打印日志）
    epochs = 100
    best_bleu = 0.0
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # 每个 epoch 重置采样器（确保打乱顺序）
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        start_time = time.time()
        batch_idx = 0

        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            batch = Batch(src, tgt, tokenizer, device)
            output = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(output, batch.trg_y, batch.ntokens)
            total_loss += loss

            # 仅主进程打印进度
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item()/batch.ntokens:.4f}")
            batch_idx += 1

        # 同步所有进程的损失（可选，用于监控全局损失）
        total_loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)  # 求和所有进程的损失
        avg_loss = total_loss_tensor.item() / (world_size * len(data_loader.dataset))

        # 仅主进程打印 epoch 总结
        if rank == 0:
            print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")

        # 仅主进程评估并保存模型
        if rank == 0 and epoch % 5 == 0:
            test_indices = random.sample(range(len(tokenizer.test)), 100)
            bleu = compute_bleu4(tokenizer, test_indices, model, device)
            print(f"BLEU-4 Score: {bleu:.2f}")
            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.module.state_dict(), f"{model_save_path}model_epoch{epoch}_bleu{bleu:.2f}.pt")  # 注意取 model.module
                print(f"Saved best model with BLEU score: {best_bleu:.2f}")

    # 最后销毁进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    # 分布式训练启动命令示例（假设使用 4 卡）：
    # torchrun --nproc_per_node=4 --master_port=29500 script.py
    en_path = r'sample-submission-version/TM-training-set/english.txt'
    ch_path = r'sample-submission-version/TM-training-set/chinese.txt'
    train(en_path, ch_path)
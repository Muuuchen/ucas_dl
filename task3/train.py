import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import time

# 数据准备
def read_data():
    '''读入数据'''
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()  # index to word
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)  # 转为torch.Tensor
    return ix2word, word2ix, data

def prepare_data(data):
    '''将数据平整为一维并滤除空格数据'''
    data = data.view(-1)
    data = data[data != word2ix['</s>']]
    return data

# 数据集类
class PoetryDataSet(Dataset):
    def __init__(self, data):
        self.seq_len = 48  # 序列长度
        self.data = prepare_data(data).long()

    def __len__(self):
        '''数据集样本批数'''
        return len(self.data) // self.seq_len

    def __getitem__(self, index):
        start_idx = index * self.seq_len
        end_idx = (index + 1) * self.seq_len
        text = self.data[start_idx: end_idx]
        label = self.data[start_idx + 1: end_idx + 1]
        return text, label

# 模型定义
LSTM_LAYERS = 3  # LSTM层数

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
                           num_layers=LSTM_LAYERS, 
                           batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        
        if hidden is None:
            h_0 = input.data.new(LSTM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(LSTM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
            hidden = (h_0, c_0)
        
        output, hidden = self.lstm(embeds, hidden)
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden

# 训练函数
def train(model, train_loader, val_loader, epochs, device, lr, scheduler_kwargs):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch, (text, label) in enumerate(train_loader):
            text = text.to(device)
            label = label.to(device).view(-1)
            
            optimizer.zero_grad()
            output, _ = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for text, label in val_loader:
                text = text.to(device)
                label = label.to(device).view(-1)
                
                output, _ = model(text)
                loss = criterion(output, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        scheduler.step()
    
    end_time = time.time()
    print(f"训练完成! 总耗时: {end_time - start_time:.2f}秒")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
    
    return model

# 生成诗词函数
def generate(model, start_words, ix2word, word2ix, max_gen_len=128, device='cpu'):
    results = list(start_words)
    input = torch.tensor([word2ix['<START>']]).unsqueeze(0).to(device)
    hidden = None
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # 首句生成
        for word in start_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)
        
        # 续写诗词
        for i in range(max_gen_len - len(start_words)):
            output, hidden = model(input, hidden)
            top_index = output.data[0].topk(1)[1][0].item()
            word = ix2word[top_index]
            
            if word == '<EOP>':
                break
                
            results.append(word)
            input = input.data.new([top_index]).view(1, 1)
    
    return ''.join(results)

# 藏头诗生成
def gen_acrostic(model, start_chars, ix2word, word2ix, max_line_len=12, device='cpu'):
    poem = []
    hidden = None
    input = torch.tensor([word2ix['<START>']]).unsqueeze(0).to(device)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for char in start_chars:
            line = [char]
            # 输入藏头字
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[char]]).view(1, 1)
            
            # 生成一行诗
            count = 0
            while True:
                output, hidden = model(input, hidden)
                top_index = output.data[0].topk(1)[1][0].item()
                word = ix2word[top_index]
                
                if word in ['，', '。', '！', '？', '<EOP>']:
                    if count % 2 == 0:
                        word = '，'
                    else:
                        word = '。'
                        line.append(word)
                        break
                
                line.append(word)
                input = input.data.new([top_index]).view(1, 1)
                count += 1
                
                if len(line) >= max_line_len:
                    break
            
            poem.append(''.join(line))
    
    return '\n'.join(poem)

if __name__ == "__main__":
    # 参数设置
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 512
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 0.001
    VAL_RATIO = 0.1  # 验证集比例
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    ix2word, word2ix, data = read_data()
    print(f"数据集形状: {data.shape}")
    
    # 准备数据集
    dataset = PoetryDataSet(data)
    total_size = len(dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    vocab_size = len(word2ix)
    model = PoetryModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    
    # 训练模型
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=device,
        lr=LR,
        scheduler_kwargs={'step_size': 5, 'gamma': 0.5}
    )
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_model.pth')
    
    # 测试生成功能
    print("\n测试诗词生成:")
    print("自由生成:")
    poem = generate(trained_model, "春风", ix2word, word2ix, device=device)
    print(poem)
    
    print("\n藏头诗生成:")
    acrostic_poem = gen_acrostic(trained_model, "深度学习", ix2word, word2ix, device=device)
    print(acrostic_poem)
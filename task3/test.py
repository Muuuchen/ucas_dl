
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import numpy as np
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import random
import os
import json
from PIL import Image
from torch.utils.data import DataLoader

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据准备函数
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
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
                           num_layers=3, 
                           batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            hidden = (h_0, c_0)
        
        output, hidden = self.lstm(embeds, hidden)
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden

# 测试函数
def evaluate_model(model, test_loader, device, criterion):
    """评估模型在测试集上的性能"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    # 存储预测结果用于后续分析
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for text, label in test_loader:
            text = text.to(device)
            label = label.to(device).view(-1)
            
            output, _ = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == label).sum().item()
            total_tokens += label.size(0)
            
            # 存储预测和目标
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"测试集评估结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"困惑度: {perplexity:.2f}")
    
    return avg_loss, accuracy, perplexity, all_predictions, all_targets

def generate_poem(model, start_words, ix2word, word2ix, max_gen_len=128, device='cpu'):
    """生成一首完整的古诗"""
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
            probs = torch.softmax(output[0], dim=0)
            top_index = torch.multinomial(probs, 1).item()
            word = ix2word[top_index]
            
            if word == '<EOP>':
                break
                
            results.append(word)
            input = input.data.new([top_index]).view(1, 1)
    
    return ''.join(results)

def gen_acrostic(model, start_chars, ix2word, word2ix, max_line_len=12, device='cpu'):
    """生成藏头诗"""
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
                probs = torch.softmax(output[0], dim=0)
                top_index = torch.multinomial(probs, 1).item()
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

def generate_poetry_report(model, ix2word, word2ix, test_loader, device):
    """生成全面的古诗模型测试报告"""
    # 创建输出目录
    os.makedirs('poetry_report', exist_ok=True)
    
    # 1. 评估模型性能
    criterion = nn.CrossEntropyLoss()
    avg_loss, accuracy, perplexity, predictions, targets = evaluate_model(
        model, test_loader, device, criterion
    )
    
    # 2. 生成示例古诗
    start_words_list = ['春风', '明月', '山水', '江南', '秋风',      "湖光秋月两相和",
        "春风又绿江南岸",
        "月落乌啼霜满天",
        "两个黄鹂鸣翠柳"]
    generated_poems = []
    
    print("\n示例古诗生成:")
    for start_words in start_words_list:
        poem = generate_poem(model, start_words, ix2word, word2ix, device=device)
        generated_poems.append(poem)
        print(f"以 '{start_words}' 开头:")
        print(poem)
        print()
    
    # 3. 生成藏头诗
    acrostic_phrases = ['人工智能', '深度学习', '自然语言', '诗情画意']
    acrostic_poems = []
    
    print("\n藏头诗生成:")
    for phrase in acrostic_phrases:
        poem = gen_acrostic(model, phrase, ix2word, word2ix, device=device)
        acrostic_poems.append(poem)
        print(f"藏头 '{phrase}':")
        print(poem)
        print()
    
    # 4. 生成随机古诗
    random_poems = []
    print("\n随机古诗生成:")
    for i in range(3):
        # 随机选择1-3个起始字
        start_len = random.randint(1, 3)
        start_words = ''.join(random.choice(list(ix2word.values())) for _ in range(start_len))
        # 过滤特殊字符
        start_words = ''.join(c for c in start_words if c not in ['<START>', '<EOP>', '，', '。', '</s>'])
        
        if not start_words:  # 防止空字符串
            start_words = random.choice(['春', '夏', '秋', '冬'])
        
        poem = generate_poem(model, start_words, ix2word, word2ix, device=device)
        random_poems.append(poem)
        print(f"随机起始 '{start_words}':")
        print(poem)
        print()
  

if __name__ == "__main__":
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    ix2word, word2ix, data = read_data()
    print(f"数据集形状: {data.shape}")
    
    # 准备测试集
    dataset = PoetryDataSet(data)
    # 使用10%的数据作为测试集
    test_size = int(len(dataset) * 0.1)
    test_indices = torch.randperm(len(dataset))[:test_size]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    vocab_size = len(word2ix)
    model = PoetryModel(vocab_size, embedding_dim=128, hidden_dim=512)
    
    # 加载训练好的模型
    model_path = 'best_model.pth'  # 或 'final_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    else:
        print(f"模型文件 {model_path} 不存在，请先训练模型")
        exit()
    
    model = model.to(device)
    
    # 生成全面的测试报告
    generate_poetry_report(model, ix2word, word2ix, test_loader, device)
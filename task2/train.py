import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transform_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 分布式训练在不同的rank上需要不同的sampler
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                         shuffle=False, num_workers=16,
                                         sampler=train_sampler)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_valid)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                        shuffle=False, num_workers=16,
                                        sampler=test_sampler)

class MHA(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embedim = self.embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self,x:torch.Tensor):
        B, S, E = x.shape
        qkv =self.qkv_proj(x).reshape(B,S,3,self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B num_head, S, E
        attn = (q @ k.transpose(-1, -2)) *self.scale # B, num_heads, S,S 
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B,S,E)
        x = self.out_proj(x)
        return x
class Transformer(nn.Module):
    def __init__(self,dim,
                 depth,
                 num_heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHA(dim, embed_dim=num_heads*dim_head ,num_heads=num_heads),
                MLP(dim,mlp_dim, dropout=dropout)
            ]))
            
    def forward(self,x):
        for attn, mlp in self.layers:
            x = x + attn(self.norm(x))
            x = x + mlp(self.norm(x))
        return self.norm(x)  
    
    
class MLP(nn.Module):
    def __init__(self, dim,hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
    def forward(self,x):
        x = self.net(x)
        return x

class ViTForCIFAR10(nn.Module):
    def __init__(self, *,
                 image_size:Tuple[int, int],
                 patch_size:Tuple[int,int],
                 num_classe:int,
                 dim,
                 depth,
                 num_heads,
                 mlp_dim,
                 pool = 'cls',
                 channels=3,
                 dim_head = 64,
                 dropout=0.,
                 embed_dropout = 0.,
                 ):
        super(ViTForCIFAR10, self).__init__()
        image_height, image_width = image_size
        self.patch_height, self.patch_width = patch_size
        assert image_height % self.patch_height == 0, "image_height must be divisible by patch_height"
        assert image_width % self.patch_height == 0, "image_weight must be divisible by patch_height"
        self.num_patch_height  = image_height // self.patch_height
        self.num_patch_width = image_width // self.patch_width
        self.num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = channels*self.patch_height*self.patch_width
        self.channels = channels
        assert pool in ['cls', 'mean'], "pool must be either cls or mean"
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(embed_dropout)

        self.transformer = Transformer(dim=dim,
                                       depth=depth,
                                        num_heads=num_heads,
                                        dim_head=dim_head,
                                        mlp_dim=mlp_dim,
                                        dropout=dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim,num_classe)


    def forward(self,x):
        batch_size, C, H,W = x.shape
        x = x.reshape(
            batch_size,C,  
            self.num_patch_height, self.patch_height, 
            self.num_patch_width,self.patch_width).permute(
                0,2,4,3,5,1).reshape(
                    batch_size,
                    self.num_patch_height*self.num_patch_width,
                    self.patch_height*self.patch_width*C
                )

        b,n,e=x.shape
        cls_tokens = torch.repeat_interleave(cls_tokens, batch_size, 0)
        
        self.to_patch_embedding(x)
        
model = ViTForCIFAR10().to(device)
model = DDP(model, device_ids=[local_rank])
model = torch.compile(model) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler()  

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'): 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()  
        scaler.update()  
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    train_loss = running_loss / len(trainloader)
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {train_loss:.3f} | Acc: {acc:.3f}%')


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Test Loss: {test_loss:.3f} | Acc: {acc:.3f}%')
    return acc


best_acc = 0
for epoch in range(50):
    train(epoch)
    acc = test(epoch)
    
    if acc > best_acc:
        print('Saving best model...')
        torch.save(model.state_dict(), 'cifar10_vit.pth')
        best_acc = acc
        
    if best_acc > 80: 
        print(f'Reached target accuracy of {best_acc}%')
        break

print(f'Best test accuracy: {best_acc}%')
dist.destroy_process_group()  


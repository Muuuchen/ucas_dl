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

class ViTForCIFAR10(nn.Module):
    def __init__(self):
        super(ViTForCIFAR10, self).__init__()
        self.vit = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=6,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=512,
            num_classes=10
        )
    
    def forward(self, x):
        return self.vit(x)

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


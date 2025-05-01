import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
import os
from sklearn.metrics import confusion_matrix

# Define the same model architecture as in train.py
class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, x: torch.Tensor):
        B, S, E = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, S, E)
        x = self.out_proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHA(dim, num_heads=num_heads),
                MLP(dim, mlp_dim, dropout=dropout)
            ]))
            
    def forward(self, x):
        for attn, mlp in self.layers:
            x = x + attn(self.norm(x))
            x = x + mlp(self.norm(x))
        return self.norm(x)

class ViTForCIFAR10(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classe, dim=768, depth=8,
                 num_heads=12, mlp_dim=2048, pool='cls', channels=3, dim_head=64,
                 dropout=0., embed_dropout=0.):
        super().__init__()
        image_height, image_width = image_size
        self.patch_height, self.patch_width = patch_size
        assert image_height % self.patch_height == 0
        assert image_width % self.patch_width == 0
        self.num_patch_height = image_height // self.patch_height
        self.num_patch_width = image_width // self.patch_width
        self.num_patches = self.num_patch_height * self.num_patch_width
        patch_dim = channels * self.patch_height * self.patch_width
        
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(embed_dropout)
        self.transformer = Transformer(dim, depth, num_heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classe)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = x.reshape(batch_size, C, self.num_patch_height, self.patch_height,
                     self.num_patch_width, self.patch_width).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(batch_size, self.num_patches, -1)
        x = self.to_patch_embedding(x)
        b, n, e = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path, device):
    """Load the trained model"""
    model = ViTForCIFAR10(
        image_size=(224, 224), 
        patch_size=(16, 16), 
        num_classe=10
    ).to(device)
    
    # Load model weights
    if os.path.exists(model_path):
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Fix the keys by removing the _orig_mod.module. prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            # Handle the case where keys have "_orig_mod.module." prefix
            if k.startswith('_orig_mod.module.'):
                name = k[len('_orig_mod.module.'):]
                new_state_dict[name] = v
            # Handle the case where keys have just "module." prefix
            elif k.startswith('module.'):
                name = k[len('module.'):]
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        # Load the fixed state dict
        model.load_state_dict(new_state_dict)
        print(f"Model successfully loaded from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found")
        return None
    
    model.eval()
    return model

def visualize_sample_predictions(model, dataloader, device, num_samples=20):
    """Visualize sample predictions with true and predicted labels"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get a batch of test images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Get predictions
    with torch.no_grad():
        sample_images = sample_images.to(device)
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    sample_images = sample_images.cpu()
    predicted = predicted.cpu()
    sample_labels = sample_labels.cpu()
    
    # Create a grid to display the results
    fig = plt.figure(figsize=(15, 12))
    rows = 4
    cols = 5
    
    # Inverse normalization for better visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Loop through samples and add to plot
    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i+1)
        
        # Convert tensor to numpy and transpose to (H,W,C)
        img = sample_images[i].numpy().transpose(1, 2, 0)
        
        # Inverse normalization
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot the image
        ax.imshow(img)
        
        # Set the title with true and predicted labels
        true_label = classes[sample_labels[i]]
        pred_label = classes[predicted[i]]
        title_color = 'green' if predicted[i] == sample_labels[i] else 'red'
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                     color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_predictions.png', dpi=200)
    plt.close()
    print("Sample predictions visualization saved as 'cifar10_predictions.png'")
    return fig

def plot_confusion_matrix(model, dataloader, device):
    """Create and visualize confusion matrix"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    y_pred = []
    y_true = []
    
    # Collect predictions
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualize
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig('cifar10_confusion_matrix.png', dpi=200)
    plt.close()
    print("Confusion matrix saved as 'cifar10_confusion_matrix.png'")
    
    return cm

def plot_class_accuracy(confusion_matrix):
    """Plot per-class accuracy"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Calculate per-class accuracy
    class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, class_accuracy * 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cifar10_class_accuracy.png', dpi=200)
    plt.close()
    print("Per-class accuracy plot saved as 'cifar10_class_accuracy.png'")

def plot_top_misclassifications(model, dataloader, device, num_examples=10):
    """Find and visualize examples with highest confidence misclassifications"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Storage for misclassified examples
    misclassified = []
    
    # Find misclassified examples with probabilities
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Finding misclassifications"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            max_probs, predicted = torch.max(probabilities, 1)
            
            # Find misclassified examples
            misclassified_mask = (predicted != labels)
            
            if misclassified_mask.sum() > 0:
                misclassified_indices = misclassified_mask.nonzero().squeeze()
                
                # Handle single misclassification case
                if misclassified_indices.dim() == 0:
                    misclassified_indices = misclassified_indices.unsqueeze(0)
                
                for idx in misclassified_indices:
                    i = idx.item()
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true': labels[i].item(),
                        'pred': predicted[i].item(),
                        'prob': max_probs[i].item()
                    })
    
    # Sort by confidence (probability) and get top examples
    misclassified.sort(key=lambda x: x['prob'], reverse=True)
    top_misclassified = misclassified[:num_examples]
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    
    # Inverse normalization for better visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i, example in enumerate(top_misclassified):
        if i >= num_examples:
            break
            
        # Get image and convert
        img = example['image'].numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(f"True: {classes[example['true']]}\nPred: {classes[example['pred']]}\nProb: {example['prob']:.2f}", 
                       color='red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('top_misclassifications.png', dpi=200)
    plt.close()
    print("Top misclassifications saved as 'top_misclassifications.png'")

def plot_training_metrics():
    """Load and visualize the training metrics from training_metrics.png"""
    # Check if the file exists
    file_path = 'training_metrics.png'
    if os.path.exists(file_path):
        # Load the image
        img = plt.imread(file_path)
        
        # Display it
        plt.figure(figsize=(12, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Training Metrics")
        plt.savefig('training_metrics_display.png', dpi=200)
        plt.close()
        print("Training metrics visualization saved as 'training_metrics_display.png'")
    else:
        print(f"Warning: Training metrics file '{file_path}' not found.")

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'cifar10_vit.pth'
    model = load_model(model_path, device)
    
    if model is None:
        return
    
    # Prepare test dataset
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=4)
    
    # Evaluate model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating model"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Generate visualizations
    visualize_sample_predictions(model, testloader, device)
    
    cm = plot_confusion_matrix(model, testloader, device)
    
    plot_class_accuracy(cm)
    
    plot_top_misclassifications(model, testloader, device)
    
    # Display training metrics from the saved image
    plot_training_metrics()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
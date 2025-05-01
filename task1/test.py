import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the same model architecture as in train.py to ensure compatibility
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class MNISTResNet(nn.Module):
    def __init__(self):
        super(MNISTResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(128, 2, stride=1)
        self.layer2 = self._make_layer(256, 2, stride=2)
        self.layer3 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to display images with predictions
def display_sample_predictions(model, test_loader, num_samples=25):
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    sample_images = sample_images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    sample_images = sample_images.cpu()
    predicted = predicted.cpu()
    sample_labels = sample_labels.cpu()
    
    # Create a grid to display the results
    fig = plt.figure(figsize=(15, 12))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    # Loop through samples and add to plot
    for i, idx in enumerate(range(len(sample_images))):
        if i >= num_samples:
            break
            
        # Plot the image
        ax = fig.add_subplot(rows, cols, i+1)
        img = sample_images[idx].squeeze().numpy()
        # Denormalize the image for better visualization
        img = (img * 0.3081) + 0.1307
        ax.imshow(img, cmap='gray')
        
        # Set the title with true and predicted labels
        title_color = 'green' if predicted[idx] == sample_labels[idx] else 'red'
        ax.set_title(f'True: {sample_labels[idx]}\nPred: {predicted[idx]}', 
                     color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_classification_results.png', dpi=200)
    plt.show()

# Function to create a confusion matrix visualization
def plot_confusion_matrix(model, test_loader):
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    confusion_mtx = np.zeros((10, 10), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion_mtx[true_label][pred_label] += 1
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations to the confusion matrix
    thresh = confusion_mtx.max() / 2.
    for i in range(confusion_mtx.shape[0]):
        for j in range(confusion_mtx.shape[1]):
            plt.text(j, i, format(confusion_mtx[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_mtx[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('mnist_confusion_matrix.png', dpi=200)
    plt.show()

# Main function
def main():
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations for test data (same as in training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the test dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize the model
    model = MNISTResNet().to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval()
    
    # Display sample predictions
    display_sample_predictions(model, test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader)
    
    # Calculate overall accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Overall Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
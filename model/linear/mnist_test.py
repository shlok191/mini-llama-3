import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from linear import Linear
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets

# Check CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),  # MNIST images are 28x28
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 classes for digits 0-9
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def train():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 1

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'

    datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
    ]

    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model and move to GPU
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if(inputs.shape[0] < 64):
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss/total,
                'acc': 100.*correct/total
            })

        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                
                if(inputs.shape[0] < 64):
                    continue
            
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}:')
        print(f'Training Accuracy: {100.*correct/total:.2f}%')
        print(f'Validation Accuracy: {100.*val_correct/val_total:.2f}%')
        print('-' * 50)

    print('Training finished!')
    return model

if __name__ == "__main__":
    model = train()
    # Save model if needed
    torch.save(model.state_dict(), 'mnist_mlp.pth')
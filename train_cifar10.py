import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models.retinalDN.retinalDN import RetinalDN
from models import ResNet18_CIFAR10

# Define transformations for training and validation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(size=(32, 32), padding=4),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 dataset with data augmentation
train_dataset = torchvision.datasets.CIFAR10(root='/kaggle/working/data', train=True, download=True, transform=transform_train)
val_dataset = torchvision.datasets.CIFAR10(root='/kaggle/working/data', train=False, download=True, transform=transform_val)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
valloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Custom loss function
def custom_loss_function(outputs, targets, images, retina_features=None, noisy_outputs=None, alpha=0.0, beta=1, is_training=True):
    ce_loss = criterion(outputs, targets)
    
    additional_loss = torch.tensor(0.0).to(device)
    noise_stability_loss = torch.tensor(0.0).to(device)
    
    if is_training:
        images.retain_grad()
        ce_loss.backward(retain_graph=True)
        gradients = images.grad
        gradients = F.normalize(gradients, p=2, dim=1)
        retina_features = retina_features.unsqueeze(1)  # Add a channel dimension
        retina_features = retina_features.repeat(1, 3, 1, 1)  # Repeat across the channel dimension to match gradients
        
        combined = gradients * retina_features
        additional_loss = combined.norm(p=2)
        noise_stability_loss = F.mse_loss(outputs, noisy_outputs)
        
        total_loss = ce_loss + alpha * additional_loss + beta * noise_stability_loss
    else:
        total_loss = ce_loss
    
    return total_loss, ce_loss, additional_loss, noise_stability_loss

# Helper function to convert RGB images to grayscale
def rgb_to_grayscale(images):
    return images.mean(dim=1, keepdim=True)

# Adjust learning rate function
def adjust_learning_rate(optimizer, epoch, initial_lr=0.01):
    if epoch >= 150:
        lr = initial_lr * 0.0001
    elif epoch >= 100:
        lr = initial_lr * 0.001
    else:
        lr = initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18_CIFAR10().to(device)
retinal_model = RetinalDN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Define training loop
num_epochs = 160

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, 0.01)
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        inputs.requires_grad = True
        outputs = model(inputs)
        
        noise = torch.randn_like(inputs) * 0.2
        noisy_inputs = inputs + noise
        noisy_outputs = model(noisy_inputs)
        
        grayscale_inputs = rgb_to_grayscale(inputs)
        retinal_features = retinal_model(grayscale_inputs)
        
        total_loss, ce_loss, additional_loss, noise_stability_loss = custom_loss_function(
            outputs, labels, inputs, retinal_features, noisy_outputs, alpha=0.0, beta=1, is_training=True
        )
        total_loss.backward()
        optimizer.step()
    


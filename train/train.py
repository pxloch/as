import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.rgb2thermal import RGB2Thermal
import os
from PIL import Image
from utils import PairedDataset

# Hyperparameters
batch_size = 8
lr = 1e-4
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_dataset = PairedDataset("dataset/train/rgb", "dataset/train/thermal", transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = RGB2Thermal().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.L1Loss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for rgb, thermal in train_loader:
        rgb, thermal = rgb.to(device), thermal.to(device)
        output = model(rgb)
        loss = loss_fn(output, thermal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "rgb2thermal.pth")

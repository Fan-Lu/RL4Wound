####################################################
# Description: LSTM autoencoder Image Implementation
# Version: V0.0.1
# Author: Sebastian Osorio @ UCSC
# Data: 2024-8-6
####################################################

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
from PIL import Image

"""
This is the image loader dataset. 

First I will establish the transformer.
Convert image to tensor and then normalize the pixel values

We also want to group the images with the same value. 
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    # I will come back to this and make it work but for now it does not
    # seem to work properly. 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def construct_image_paths(base_dir, img_names):
    sub_dirs = [str(i) for i in range(16)]  # Subdirectories 0 to 15
    img_paths = []
    
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            print(f"Subdirectory does not exist: {sub_dir_path}")
            continue  # Skip if subdirectory does not exist
        
        for img_name in img_names:
            # Construct full image path
            full_path = os.path.join(sub_dir_path, img_name)
            if os.path.isfile(full_path):
                img_paths.append(full_path)  # Store path
            
    
    return img_paths

train_csv = pd.read_csv('/Users/bearcbass/RL4Wound/data/MouseData/all_train_imgs.csv')
val_csv = pd.read_csv('/Users/bearcbass/RL4Wound/data/MouseData/all_test_imgs.csv')

# Filter rows where WNum is 1
train_imgs = train_csv[train_csv['WNum'] == 1]['ImNa'].tolist()
val_imgs = val_csv[val_csv['WNum'] == 1]['ImNa'].tolist()

# Define the path
train_img_dir = '/Users/bearcbass/RL4Wound/data/MouseData/train'
val_img_dir = '/Users/bearcbass/RL4Wound/data/MouseData/val'

train_img_paths = construct_image_paths(train_img_dir, train_imgs)
val_img_paths = construct_image_paths(val_img_dir, val_imgs)

class ImagesDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # print('This are the images paths in the constructor')
        # print(img_path)
        
        # Debugging print statement
        if not os.path.isfile(img_path):
            print(f"File not found in __getitem__: {img_path}")
            raise FileNotFoundError(f"No such file: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    
# Create datasets
train_dataset = ImagesDataset(img_paths = train_img_paths , transform=transform)
val_dataset = ImagesDataset(img_paths = val_img_paths, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def show_images(images, nrow=4, ncol=4):
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 10))
    axes = axes.flatten()
    
    for img, ax in zip(images, axes):
        ax.imshow(img.permute(1, 2, 0))  # Reorder dimensions to (H, W, C)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# Get a batch of training images
data_iter = iter(train_loader)
images = next(data_iter)

# Visualize the batch of images
# show_images(images[:16])  # Show the first 16 images in the batch

class LSTMEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(LSTMEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3,64, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(256 * 19 * 19, latent_dim, batch_first=True)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = self.flatten(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)

        return x

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(LSTMDecoder, self).__init__()
        
        self.lstm = nn.LSTM(latent_dim, 256 * 19 * 19, batch_first=True)
        
        self.unflatten = nn.Unflatten(1, (256, 19, 19))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.unflatten(x)
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = self.deconv3(x)

        return x
    
class LSTMParameterPredictor(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(LSTMParameterPredictor, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(latent_dim)
        self.decoder = LSTMDecoder(latent_dim)
        self.param_predictor = LSTMParameterPredictor(latent_dim, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        params_pred = self.param_predictor(latent)
        return reconstructed, params_pred
    
# Latent dimension can be changed
latent_dim = 100
num_classes = 4
learning_rate = 0.001

model = LSTMAutoencoder(latent_dim, num_classes)
criterion_recon = nn.MSELoss()
criterion_params = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss_recon = 0.0
    epoch_loss_params = 0.0

    for batch_data, _ in train_loader:
        optimizer.zero_grad()

        reconstructed, params_pred = model(batch_data)

        loss_recon = criterion_recon(reconstructed, batch_data)

        loss = loss_recon

        loss.backward()
        optimizer.step()

        epoch_loss_recon += loss_recon.item()

    model.eval()
    val_loss_recon = 0.0
    with torch.no_grad():
        for val_data, _ in val_loader:
            val_reconstructed, _ = model(val_data)
            val_loss_recon += criterion_recon(val_reconstructed, val_data).item()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Recon Loss: {epoch_loss_recon:.4f}, Val Recon Loss: {val_loss_recon:.4f}")
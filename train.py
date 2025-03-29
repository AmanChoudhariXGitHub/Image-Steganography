import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SteganographyDataset
from model import SteganoNetwork
from metrics import calculate_psnr, calculate_ssim, calculate_payload_capacity, test_robustness
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
num_epochs = 100
learning_rate = 0.001
beta = 0.75  # Weight for the stego image loss

# Create model
model = SteganoNetwork().to(device)

# Loss function
mse_loss = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = SteganographyDataset(root_dir='data/train', transform=transform)
val_dataset = SteganographyDataset(root_dir='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training function
def train_model():
    # Create directory for saving models
    os.makedirs('saved_models', exist_ok=True)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_psnrs_stego = []
    val_psnrs_revealed = []
    val_ssims_stego = []
    val_ssims_revealed = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for i, (secret_imgs, cover_imgs) in enumerate(train_loader):
            secret_imgs = secret_imgs.to(device)
            cover_imgs = cover_imgs.to(device)
            
            # Forward pass
            prepared_secret, stego_imgs, revealed_imgs = model(secret_imgs, cover_imgs)
            
            # Calculate losses
            loss_stego = mse_loss(stego_imgs, cover_imgs)
            loss_revealed = mse_loss(revealed_imgs, secret_imgs)
            
            # Combined loss
            loss = beta * loss_stego + (1 - beta) * loss_revealed
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_stego = 0.0
        val_psnr_revealed = 0.0
        val_ssim_stego = 0.0
        val_ssim_revealed = 0.0
        
        with torch.no_grad():
            for secret_imgs, cover_imgs in val_loader:
                secret_imgs = secret_imgs.to(device)
                cover_imgs = cover_imgs.to(device)
                
                # Forward pass
                prepared_secret, stego_imgs, revealed_imgs = model(secret_imgs, cover_imgs)
                
                # Calculate losses
                loss_stego = mse_loss(stego_imgs, cover_imgs)
                loss_revealed = mse_loss(revealed_imgs, secret_imgs)
                loss = beta * loss_stego + (1 - beta) * loss_revealed
                
                val_loss += loss.item()
                
                # Calculate metrics
                val_psnr_stego += calculate_psnr(cover_imgs, stego_imgs)
                val_psnr_revealed += calculate_psnr(secret_imgs, revealed_imgs)
                val_ssim_stego += calculate_ssim(cover_imgs, stego_imgs)
                val_ssim_revealed += calculate_ssim(secret_imgs, revealed_imgs)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_psnr_stego /= len(val_loader)
        val_psnrs_stego.append(val_psnr_stego)
        
        val_psnr_revealed /= len(val_loader)
        val_psnrs_revealed.append(val_psnr_revealed)
        
        val_ssim_stego /= len(val_loader)
        val_ssims_stego.append(val_ssim_stego)
        
        val_ssim_revealed /= len(val_loader)
        val_ssims_revealed.append(val_ssim_revealed)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Stego PSNR: {val_psnr_stego:.2f} dB, Revealed PSNR: {val_psnr_revealed:.2f} dB')
        print(f'Stego SSIM: {val_ssim_stego:.4f}, Revealed SSIM: {val_ssim_revealed:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print("Saved best model!")
        
        # Save checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'saved_models/checkpoint_epoch_{epoch+1}.pth')
    
    # Plot and save training curves
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(val_psnrs_stego, label='Stego PSNR')
    plt.plot(val_psnrs_revealed, label='Revealed PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('PSNR Values')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_ssims_stego, label='Stego SSIM')
    plt.plot(val_ssims_revealed, label='Revealed SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM Values')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    train_model()


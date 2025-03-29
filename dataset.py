import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SteganographyDataset(Dataset):
    """Dataset for steganography training with pairs of secret and cover images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
                Expected to have two subdirectories: 'secret' and 'cover'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.secret_dir = os.path.join(root_dir, 'secret')
        self.cover_dir = os.path.join(root_dir, 'cover')
        
        self.secret_imgs = [f for f in os.listdir(self.secret_dir) if os.path.isfile(os.path.join(self.secret_dir, f))]
        self.cover_imgs = [f for f in os.listdir(self.cover_dir) if os.path.isfile(os.path.join(self.cover_dir, f))]
        
        # If there are more cover images than secret images, we'll randomly sample cover images
        # If there are more secret images than cover images, we'll randomly sample secret images
        self.length = min(len(self.secret_imgs), len(self.cover_imgs))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # For training, we randomly pair secret and cover images
        secret_idx = idx % len(self.secret_imgs)
        cover_idx = idx % len(self.cover_imgs)
        
        secret_img_name = os.path.join(self.secret_dir, self.secret_imgs[secret_idx])
        cover_img_name = os.path.join(self.cover_dir, self.cover_imgs[cover_idx])
        
        secret_image = Image.open(secret_img_name).convert('RGB')
        cover_image = Image.open(cover_img_name).convert('RGB')
        
        if self.transform:
            secret_image = self.transform(secret_image)
            cover_image = self.transform(cover_image)
        
        return secret_image, cover_image


import torch
import torch.nn as nn
import torch.nn.functional as F

class PrepNetwork(nn.Module):
    """Preparation Network for the secret image"""
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(50, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class HidingNetwork(nn.Module):
    """Network for hiding the prepared secret image into the cover image"""
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(50, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class RevealNetwork(nn.Module):
    """Network for revealing the secret image from the stego image"""
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(50, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class SteganoNetwork(nn.Module):
    """Complete steganography network combining preparation, hiding, and revealing"""
    def __init__(self):
        super(SteganoNetwork, self).__init__()
        self.prep_network = PrepNetwork()
        self.hiding_network = HidingNetwork()
        self.reveal_network = RevealNetwork()
        
    def forward(self, secret, cover):
        # Prepare the secret image
        prepared_secret = self.prep_network(secret)
        
        # Concatenate the prepared secret and cover images
        concat_input = torch.cat([prepared_secret, cover], dim=1)
        
        # Hide the secret image in the cover image
        stego = self.hiding_network(concat_input)
        
        # Reveal the secret image from the stego image
        revealed = self.reveal_network(stego)
        
        return prepared_secret, stego, revealed


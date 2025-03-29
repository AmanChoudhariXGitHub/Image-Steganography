import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # If batched, calculate mean PSNR across batch
    if len(img1.shape) == 4:  # Batch of images
        psnr_values = []
        for i in range(img1.shape[0]):
            # Convert from [C,H,W] to [H,W,C] and scale to [0, 255]
            i1 = np.transpose(img1[i], (1, 2, 0)) * 255.0
            i2 = np.transpose(img2[i], (1, 2, 0)) * 255.0
            mse = np.mean((i1 - i2) ** 2)
            if mse == 0:
                psnr_values.append(100.0)
            else:
                psnr_values.append(20 * np.log10(255.0 / np.sqrt(mse)))
        return np.mean(psnr_values)
    else:
        # Convert from [C,H,W] to [H,W,C] and scale to [0, 255]
        img1 = np.transpose(img1, (1, 2, 0)) * 255.0
        img2 = np.transpose(img2, (1, 2, 0)) * 255.0
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # If batched, calculate mean SSIM across batch
    if len(img1.shape) == 4:  # Batch of images
        ssim_values = []
        for i in range(img1.shape[0]):
            # Convert from [C,H,W] to [H,W,C] and scale to [0, 255]
            i1 = np.transpose(img1[i], (1, 2, 0)) * 255.0
            i2 = np.transpose(img2[i], (1, 2, 0)) * 255.0

            # Get the smallest image dimension
            min_dim = min(i1.shape[0], i1.shape[1])  
    
            # Ensure win_size is odd and smaller than the smallest dimension
            if min_dim < 7:
                # If image is too small, use the smallest possible window size
                win_size = min(3, min_dim)
                if win_size % 2 == 0:  # Ensure odd window size
                    win_size -= 1
            else:
                win_size = 7  # Default window size
            
            # Check if window size is valid
            if win_size < 1:
                # If image is extremely small, return a placeholder value
                ssim_values.append(1.0 if np.array_equal(i1, i2) else 0.0)
            else:
                try:
                    ssim_val = ssim(i1, i2, multichannel=True, data_range=255, win_size=win_size, channel_axis=2)
                    ssim_values.append(ssim_val)
                except Exception as e:
                    # If SSIM calculation fails, use a simple similarity measure
                    mse = np.mean((i1 - i2) ** 2)
                    similarity = 1.0 / (1.0 + mse / 255.0)
                    ssim_values.append(similarity)
        
        return np.mean(ssim_values)  # Return the average SSIM across the batch
    else:
        # Convert from [C,H,W] to [H,W,C] and scale to [0, 255]
        img1 = np.transpose(img1, (1, 2, 0)) * 255.0
        img2 = np.transpose(img2, (1, 2, 0)) * 255.0
        
        # Get the smallest image dimension
        min_dim = min(img1.shape[0], img1.shape[1])
        
        # Ensure win_size is odd and smaller than the smallest dimension
        if min_dim < 7:
            win_size = min(3, min_dim)
            if win_size % 2 == 0:  # Ensure odd window size
                win_size -= 1
        else:
            win_size = 7  # Default window size
            
        # Check if window size is valid
        if win_size < 1:
            # If image is extremely small, return a placeholder value
            return 1.0 if np.array_equal(img1, img2) else 0.0
        
        try:
            return ssim(img1, img2, multichannel=True, data_range=255, win_size=win_size, channel_axis=2)
        except Exception as e:
            # If SSIM calculation fails, use a simple similarity measure
            mse = np.mean((img1 - img2) ** 2)
            return 1.0 / (1.0 + mse / 255.0)

def calculate_payload_capacity(cover_img_size, secret_img_size):
    """Calculate payload capacity in bits per pixel (bpp)"""
    # Assuming 3 channels (RGB) for both images
    cover_pixels = cover_img_size[0] * cover_img_size[1]
    secret_bits = secret_img_size[0] * secret_img_size[1] * 3 * 8  # 3 channels, 8 bits per channel
    
    # Payload capacity in bits per pixel
    bpp = secret_bits / cover_pixels
    
    return bpp

def test_robustness(model, stego_img, attacks=None):
    """Test robustness of the steganography model against various attacks"""
    if attacks is None:
        attacks = ['gaussian_noise', 'jpeg_compression', 'gaussian_blur', 'resize']
    
    results = {}
    device = next(model.parameters()).device
    
    # Make sure stego_img is a tensor on the correct device
    if not torch.is_tensor(stego_img):
        stego_img = torch.tensor(stego_img).to(device)
    else:
        stego_img = stego_img.to(device)
    
    # Add batch dimension if needed
    if len(stego_img.shape) == 3:
        stego_img = stego_img.unsqueeze(0)
    
    # Original extraction without attack
    with torch.no_grad():
        revealed_img = model.reveal_network(stego_img)
    
    results['original'] = {
        'revealed_img': revealed_img.detach().cpu()
    }
    
    # Apply attacks and test
    for attack in attacks:
        attacked_img = stego_img.clone()
        
        if attack == 'gaussian_noise':
            # Add Gaussian noise
            noise = torch.randn_like(attacked_img) * 0.05 * attacked_img.max()
            attacked_img = attacked_img + noise
            attacked_img = torch.clamp(attacked_img, 0, 1)
            
        elif attack == 'jpeg_compression':
            # Simulate JPEG compression by converting to numpy, applying compression, and back to tensor
            attacked_img_np = attacked_img.detach().cpu().numpy()
            attacked_img_compressed = []
            
            for i in range(attacked_img_np.shape[0]):
                img = (np.transpose(attacked_img_np[i], (1, 2, 0)) * 255).astype(np.uint8)
                # Apply JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Quality factor 75
                _, encimg = cv2.imencode('.jpg', img, encode_param)
                decimg = cv2.imdecode(encimg, 1)
                attacked_img_compressed.append(np.transpose(decimg, (2, 0, 1)) / 255.0)
            
            attacked_img = torch.tensor(np.array(attacked_img_compressed), dtype=torch.float32).to(device)
            
        elif attack == 'gaussian_blur':
            # Apply Gaussian blur
            kernel_size = 5
            sigma = 1.0
            attacked_img_np = attacked_img.detach().cpu().numpy()
            attacked_img_blurred = []
            
            for i in range(attacked_img_np.shape[0]):
                img = (np.transpose(attacked_img_np[i], (1, 2, 0)) * 255).astype(np.uint8)
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
                attacked_img_blurred.append(np.transpose(blurred, (2, 0, 1)) / 255.0)
            
            attacked_img = torch.tensor(np.array(attacked_img_blurred), dtype=torch.float32).to(device)
            
        elif attack == 'resize':
            # Resize down and up to simulate loss of information
            h, w = attacked_img.shape[2], attacked_img.shape[3]
            attacked_img = F.interpolate(attacked_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            attacked_img = F.interpolate(attacked_img, size=(h, w), mode='bilinear', align_corners=False)
        
        # Extract secret image after attack
        with torch.no_grad():
            revealed_img_after_attack = model.reveal_network(attacked_img)
        
        # Calculate metrics
        psnr = calculate_psnr(results['original']['revealed_img'], revealed_img_after_attack.detach().cpu())
        ssim_val = calculate_ssim(results['original']['revealed_img'], revealed_img_after_attack.detach().cpu())
        
        results[attack] = {
            'revealed_img': revealed_img_after_attack.detach().cpu(),
            'psnr': psnr,
            'ssim': ssim_val
        }
    
    return results


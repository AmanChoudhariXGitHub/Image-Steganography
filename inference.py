import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse
from model import SteganoNetwork
from metrics import calculate_psnr, calculate_ssim, calculate_payload_capacity, test_robustness
from skimage.transform import resize

def load_image(image_path, transform=None):
    """Load an image and apply transformations"""
    image = Image.open(image_path).convert('RGB')  # Ensure PIL Image
    if transform:
        return transform(image)  # Apply transform and return tensor
    return image  # Return PIL Image if no transform

def resize_image(img, min_size=7):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    w, h = img.size
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.ANTIALIAS)
    
    return img


def tensor_to_image(tensor):
    """Convert a tensor to a numpy image"""
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]  # Take the first image
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img, 0, 1)
    return img

def main():
    parser = argparse.ArgumentParser(description='Steganography Inference')
    parser.add_argument('--model', type=str, default='saved_models/best_model.pth', help='Path to the model')
    parser.add_argument('--secret', type=str, required=True, help='Path to the secret image')
    parser.add_argument('--cover', type=str, required=True, help='Path to the cover image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save the stego image')
    parser.add_argument('--test_robustness', action='store_true', help='Test robustness against attacks')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = SteganoNetwork().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Load images as PIL
    secret_img = load_image(args.secret)
    cover_img = load_image(args.cover)
    # secret_img = load_image(args.secret, transform).to(device)
    # cover_img = load_image(args.cover, transform).to(device)
    
    # Resize if necessary
    secret_img = resize_image(secret_img)
    cover_img = resize_image(cover_img)
    
    # Apply transformations (resize to 256x256, convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    secret_img = transform(secret_img).to(device)
    cover_img = transform(cover_img).to(device)
    
    # Add batch dimension
    secret_img = secret_img.unsqueeze(0)
    cover_img = cover_img.unsqueeze(0)
    
    # Perform steganography
    with torch.no_grad():
        prepared_secret, stego_img, revealed_img = model(secret_img, cover_img)
    
    # Calculate metrics
    psnr_stego = calculate_psnr(cover_img, stego_img)
    psnr_revealed = calculate_psnr(secret_img, revealed_img)
    ssim_stego = calculate_ssim(cover_img, stego_img)
    ssim_revealed = calculate_ssim(secret_img, revealed_img)
    
    # Calculate payload capacity
    payload_capacity = calculate_payload_capacity((256, 256), (256, 256))
    
    print(f"Stego PSNR: {psnr_stego:.2f} dB")
    print(f"Revealed PSNR: {psnr_revealed:.2f} dB")
    print(f"Stego SSIM: {ssim_stego:.4f}")
    print(f"Revealed SSIM: {ssim_revealed:.4f}")
    print(f"Payload Capacity: {payload_capacity:.4f} bpp")
    
    # Save stego image
    stego_img_np = tensor_to_image(stego_img)
    plt.imsave(args.output, stego_img_np)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(tensor_to_image(secret_img))
    plt.title('Secret Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(tensor_to_image(cover_img))
    plt.title('Cover Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(stego_img_np)
    plt.title(f'Stego Image (PSNR: {psnr_stego:.2f} dB, SSIM: {ssim_stego:.4f})')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(tensor_to_image(revealed_img))
    plt.title(f'Revealed Image (PSNR: {psnr_revealed:.2f} dB, SSIM: {ssim_revealed:.4f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results.png')
    
    # Test robustness if requested
    if args.test_robustness:
        print("\nTesting robustness against attacks...")
        robustness_results = test_robustness(model, stego_img)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(tensor_to_image(robustness_results['original']['revealed_img']))
        plt.title('Original Revealed')
        plt.axis('off')
        
        attack_idx = 2
        for attack, result in robustness_results.items():
            if attack == 'original':
                continue
                
            plt.subplot(2, 3, attack_idx)
            plt.imshow(tensor_to_image(result['revealed_img']))
            plt.title(f'{attack.replace("_", " ").title()}\nPSNR: {result["psnr"]:.2f} dB\nSSIM: {result["ssim"]:.4f}')
            plt.axis('off')
            attack_idx += 1
        
        plt.tight_layout()
        plt.savefig('robustness_results.png')
        
        print("Robustness test results:")
        for attack, result in robustness_results.items():
            if attack == 'original':
                continue
            print(f"  {attack.replace('_', ' ').title()}:")
            print(f"    PSNR: {result['psnr']:.2f} dB")
            print(f"    SSIM: {result['ssim']:.4f}")

if __name__ == "__main__":
    main()

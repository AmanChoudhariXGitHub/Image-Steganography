import os
import shutil
from PIL import Image
import numpy as np

# Create necessary directories
os.makedirs("data/stego", exist_ok=True)
os.makedirs("data/revealed", exist_ok=True)
os.makedirs("data/secret_store", exist_ok=True)

# Create a default secret image (a colorful pattern)
def create_default_secret():
    width, height = 256, 256
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a colorful pattern
    for y in range(height):
        for x in range(width):
            r = int(128 + 127 * np.sin(x / 30))
            g = int(128 + 127 * np.sin(y / 30))
            b = int(128 + 127 * np.sin((x + y) / 30))
            img[y, x] = [r, g, b]
    
    # Save the image
    Image.fromarray(img).save("data/default_secret.png")
    print("Created default secret image at data/default_secret.png")

# Copy a sample secret image from training data if available
def copy_sample_secret():
    try:
        if os.path.exists("data/train/secret"):
            secret_files = os.listdir("data/train/secret")
            if secret_files:
                src = os.path.join("data/train/secret", secret_files[0])
                dst = "data/default_secret.png"
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
                return True
    except Exception as e:
        print(f"Error copying sample secret: {e}")
    return False

if __name__ == "__main__":
    if not copy_sample_secret():
        create_default_secret()
    
    print("Setup complete!")

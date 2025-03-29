import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from torchvision import transforms
from model import SteganoNetwork
from metrics import calculate_psnr, calculate_ssim, calculate_payload_capacity, test_robustness

# Set page config
st.set_page_config(
    page_title="Deep Learning Image Steganography",
    page_icon="🔒",
    layout="wide"
)

# Function to load model
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SteganoNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Function to process images
def process_images(secret_img, cover_img, model, device, mode="encode", stego_img=None):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Process secret image
    if isinstance(secret_img, Image.Image):
        secret_tensor = transform(secret_img).unsqueeze(0).to(device)
    else:
        secret_tensor = None
    
    # Process cover image
    if isinstance(cover_img, Image.Image):
        cover_tensor = transform(cover_img).unsqueeze(0).to(device)
    else:
        cover_tensor = None
    
    # Process stego image if provided
    if isinstance(stego_img, Image.Image):
        stego_tensor = transform(stego_img).unsqueeze(0).to(device)
    else:
        stego_tensor = None
    
    results = {}
    
    with torch.no_grad():
        if mode == "encode" and secret_tensor is not None and cover_tensor is not None:
            # Encoding mode
            prepared_secret, stego_tensor, _ = model(secret_tensor, cover_tensor)
            
            # Calculate metrics
            psnr_stego = calculate_psnr(cover_tensor, stego_tensor)
            ssim_stego = calculate_ssim(cover_tensor, stego_tensor)
            payload_capacity = calculate_payload_capacity((256, 256), (256, 256))
            
            results = {
                "stego_img": stego_tensor,
                "psnr_stego": psnr_stego,
                "ssim_stego": ssim_stego,
                "payload_capacity": payload_capacity
            }
            
        elif mode == "decode" and stego_tensor is not None:
            # Decoding mode
            revealed_tensor = model.reveal_network(stego_tensor)
            
            results = {
                "revealed_img": revealed_tensor
            }
    
    return results

# Function to convert tensor to PIL image
def tensor_to_pil(tensor):
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]  # Take the first image
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# Function to convert PIL to bytes for download
def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return buf.getvalue()

# Main app
def main():
    st.title("Deep Learning Image Steganography")
    st.markdown("""
    This application uses deep learning with Convolutional Neural Networks (CNNs) to perform image steganography.
    You can either encode a secret image into a cover image or decode a stego image to reveal the hidden secret.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["saved_models/best_model.pth"],
        index=0
    )
    
    # Load model
    try:
        model, device = load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()
    
    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Encode", "Decode"])
    
    if mode == "Encode":
        st.header("Encode a Secret Image")
        
        # Upload images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Secret Image")
            secret_file = st.file_uploader("Upload Secret Image", type=["jpg", "jpeg", "png"])
            if secret_file is not None:
                secret_img = Image.open(secret_file).convert('RGB')
                st.image(secret_img, caption="Secret Image", use_column_width=True)
            else:
                secret_img = None
        
        with col2:
            st.subheader("Cover Image")
            cover_file = st.file_uploader("Upload Cover Image", type=["jpg", "jpeg", "png"])
            if cover_file is not None:
                cover_img = Image.open(cover_file).convert('RGB')
                st.image(cover_img, caption="Cover Image", use_column_width=True)
            else:
                cover_img = None
        
        # Process button
        if st.button("Encode Images") and secret_img is not None and cover_img is not None:
            with st.spinner("Processing..."):
                results = process_images(secret_img, cover_img, model, device, mode="encode")
                
                if "stego_img" in results:
                    st.header("Results")
                    
                    # Display stego image
                    stego_pil = tensor_to_pil(results["stego_img"])
                    st.image(stego_pil, caption=f"Stego Image", use_column_width=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("PSNR (Stego)", f"{results['psnr_stego']:.2f} dB")
                    col2.metric("SSIM (Stego)", f"{results['ssim_stego']:.4f}")
                    col3.metric("Payload Capacity", f"{results['payload_capacity']:.4f} bpp")
                    
                    # Download button
                    st.download_button(
                        label="Download Stego Image",
                        data=pil_to_bytes(stego_pil),
                        file_name="stego_image.png",
                        mime="image/png"
                    )
                    
                    # Test robustness
                    if st.checkbox("Test Robustness Against Attacks"):
                        with st.spinner("Testing robustness..."):
                            robustness_results = test_robustness(model, results["stego_img"])
                            
                            st.subheader("Robustness Test Results")
                            
                            # Create tabs for different attacks
                            tabs = st.tabs(["Original"] + [attack.replace("_", " ").title() for attack in robustness_results if attack != "original"])
                            
                            # Original tab
                            with tabs[0]:
                                revealed_original = tensor_to_pil(robustness_results["original"]["revealed_img"])
                                st.image(revealed_original, caption="Original Revealed Image", use_column_width=True)
                            
                            # Attack tabs
                            tab_idx = 1
                            for attack, result in robustness_results.items():
                                if attack == "original":
                                    continue
                                    
                                with tabs[tab_idx]:
                                    revealed_after_attack = tensor_to_pil(result["revealed_img"])
                                    st.image(revealed_after_attack, caption=f"Revealed After {attack.replace('_', ' ').title()}", use_column_width=True)
                                    
                                    col1, col2 = st.columns(2)
                                    col1.metric("PSNR", f"{result['psnr']:.2f} dB")
                                    col2.metric("SSIM", f"{result['ssim']:.4f}")
                                
                                tab_idx += 1
    
    else:  # Decode mode
        st.header("Decode a Stego Image")
        
        # Upload stego image
        stego_file = st.file_uploader("Upload Stego Image", type=["jpg", "jpeg", "png"])
        if stego_file is not None:
            stego_img = Image.open(stego_file).convert('RGB')
            st.image(stego_img, caption="Stego Image", use_column_width=True)
        else:
            stego_img = None
        
        # Process button
        if st.button("Decode Image") and stego_img is not None:
            with st.spinner("Processing..."):
                results = process_images(None, None, model, device, mode="decode", stego_img=stego_img)
                
                if "revealed_img" in results:
                    st.header("Results")
                    
                    # Display revealed image
                    revealed_pil = tensor_to_pil(results["revealed_img"])
                    st.image(revealed_pil, caption="Revealed Secret Image", use_column_width=True)
                    
                    # Download button
                    st.download_button(
                        label="Download Revealed Image",
                        data=pil_to_bytes(revealed_pil),
                        file_name="revealed_secret.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()


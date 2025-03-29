Deep Learning-Based Image Steganography
🔒 Securely hide images within images using Deep Learning and CNNs

📌 Overview
This project implements an image steganography system using Convolutional Neural Networks (CNNs). The system allows hiding a secret image inside a cover image and later extracting it with minimal loss. Unlike traditional steganography methods, this approach leverages deep learning to achieve higher imperceptibility and robustness.

🚀 Features
Encode (Hide Image): Embed a secret image within a cover image.
Decode (Extract Image): Retrieve the hidden image from the stego image.
Deep Learning-based Approach: Utilizes a CNN model for efficient encoding and decoding.
Performance Evaluation: Measures PSNR, SSIM, payload capacity, and robustness.
User-Friendly Interface: Streamlit-based UI for easy interaction.

🖥️ Technologies Used
Python (Deep Learning & Image Processing)
TensorFlow / Keras (CNN Model Training)
OpenCV (Image Preprocessing)
Streamlit (Web Interface)
NumPy & Matplotlib (Data Processing & Visualization)

📊 Performance Metrics
Peak Signal-to-Noise Ratio (PSNR): Measures image quality after encoding.
Structural Similarity Index (SSIM): Evaluates how well the stego image retains visual similarity.
Payload Capacity: Determines how much information can be hidden.
Robustness: Checks resistance against noise and distortions.

📂 Project Structure
bash
Copy
Edit
📦 Image-Stegano-CNN
 ┣ 📂 data/                  # Training data (cover & secret images)
 ┣ 📂 models/                # Saved model weights
 ┣ 📂 src/                   # Source code
 ┃ ┣ 📜 train.py             # Model training script
 ┃ ┣ 📜 inference.py         # Encoding & Decoding script
 ┃ ┣ 📜 metrics.py           # Performance evaluation functions
 ┃ ┣ 📜 utils.py             # Helper functions
 ┣ 📜 app.py                 # Streamlit Web App
 ┣ 📜 README.md              # Project documentation
 ┗ 📜 requirements.txt       # Dependencies

📌 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Train the Model (Optional)
python train.py
3️⃣ Encode an Image (Hide Secret Image)
python inference.py --secret <path_to_secret_image> --cover <path_to_cover_image>
4️⃣ Decode the Hidden Image
python inference.py --decode --stego <path_to_stego_image>
5️⃣ Run Web App
streamlit run app.py

📌 Future Enhancements
Improve model accuracy for higher PSNR & SSIM
Implement video steganography
Add adversarial training for enhanced security

📜 License
This project is open-source and licensed under the MIT License.
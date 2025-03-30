Deep Learning-Based Image Steganography
🔒 Securely hide images within images using Deep Learning and CNNs



You can access the web-app from :-
https://image-steganography-cnn.streamlit.app/



📌 Overview
This project implements an image steganography system using Convolutional Neural Networks (CNNs). The system allows hiding a secret image inside a cover image and later extracting it with minimal loss. Unlike traditional steganography methods, this approach leverages deep learning to achieve higher imperceptibility and robustness.



🚀 Features
1. Encode (Hide Image): Embed a secret image within a cover image.
2. Decode (Extract Image): Retrieve the hidden image from the stego image.
3. Deep Learning-based Approach: Utilizes a CNN model for efficient encoding and decoding.
4. Performance Evaluation: Measures PSNR, SSIM, payload capacity, and robustness.
5. User-Friendly Interface: Streamlit-based UI for easy interaction.



🖥️ Technologies Used
1. Python (Deep Learning & Image Processing)
2. TensorFlow / Keras (CNN Model Training)
3. OpenCV (Image Preprocessing)
4. Streamlit (Web Interface)
5. NumPy & Matplotlib (Data Processing & Visualization)



📊 Performance Metrics
1. Peak Signal-to-Noise Ratio (PSNR): Measures image quality after encoding.
2. Structural Similarity Index (SSIM): Evaluates how well the stego image retains visual similarity.
3. Payload Capacity: Determines how much information can be hidden.
4. Robustness: Checks resistance against noise and distortions.



📂 Project Structure

📦 Image-Steganography

 ┣ 📂 data/                  # Training data (cover & secret images)

 ┣ 📂 models/                # Saved model weights

 ┣ 📜 train.py             # Model training script

 ┣ 📜 inference.py         # Encoding & Decoding script

 ┣ 📜 metrics.py           # Performance evaluation functions

 ┣ 📜 utils.py             # Helper functions

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
1. Improve model accuracy for higher PSNR & SSIM
2. Implement video steganography
3. Add adversarial training for enhanced security



📜 License
This project is open-source and licensed under the MIT License.

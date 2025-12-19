# Face Shape Detection using EfficientNet-B4

This project is a Streamlit web application that detects a person’s face shape from an image using a deep learning model. The app supports both live camera input and image uploads, automatically detects the face, and predicts the face shape along with a confidence score.

The model is fine-tuned using EfficientNet-B4 and trained on a labeled face shape dataset containing five face shape categories.

---

## Face Shape Classes

The model predicts one of the following face shapes:
- Oval
- Round
- Oblong
- Square
- Heart

---

## Dataset

The model is trained on a Face Shape dataset with the following distribution:

- **Training data:**  
  - 1,000 images per face shape  
- **Testing data:**  
  - 200 images per face shape  

All images are organized in class-wise folders and contain clear face images suitable for classification.

---

## Model Architecture

- **Model:** EfficientNet-B4  
- **Framework:** PyTorch  
- **Training Method:** Transfer learning with a custom classification head  
- **Input Size:** 224 × 224  
- **Output:** 5 face shape classes  
- **Inference Device:** CPU  

EfficientNet-B4 is chosen for its balance between accuracy and computational efficiency, making it suitable for real-time inference in lightweight applications.

---

## How the System Works

1. The user captures an image using the camera or uploads an image.
2. OpenCV Haar Cascade is used to detect faces in the image.
3. The largest detected face is cropped for better accuracy.
4. The cropped face is resized and normalized.
5. The EfficientNet-B4 model predicts the face shape.
6. The predicted label and confidence score are displayed in the Streamlit UI.

---

## Features

- Camera and image upload support
- Automatic face detection and cropping
- Real-time face shape prediction
- Confidence score for each prediction
- Simple and interactive Streamlit interface
- Runs entirely on CPU

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-shape-detection.git
cd face-shape-detection

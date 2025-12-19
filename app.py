import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

# -----------------------
# Config
# -----------------------
DEVICE = "cpu"   # CPU only

CLASS_NAMES = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square"
]

MODEL_PATH = "best_model.pth"

# -----------------------
# Load face detector
# -----------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------
# Face detection function
# -----------------------
def detect_face(image: Image.Image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Select largest face
    x, y, w, h = sorted(
        faces, key=lambda f: f[2] * f[3], reverse=True
    )[0]

    face_crop = img[y:y+h, x:x+w]
    return Image.fromarray(face_crop)


# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    model = torchvision.models.efficientnet_b4(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------
# Transforms
# -----------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Prediction function
# -----------------------
def predict_face_shape(face_image: Image.Image):
    image = transform(face_image).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], conf.item()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Face Shape Detector", layout="centered")
st.title("Face Shape Detection")
st.write("Detect your face shape using camera or image upload")

option = st.radio(
    "Choose input method:",
    ("Use Camera", "Upload Image")
)

image = None

# -------- Camera input --------
if option == "Use Camera":
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

# -------- Upload input --------
else:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# -------- Face check + Prediction --------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Detecting face..."):
        face = detect_face(image)

    if face is None:
        st.warning("No face detected. Please upload a clear photo with a visible face.")
    else:
        st.image(face, caption="Detected Face", width=200)

        with st.spinner("Analyzing face shape..."):
            label, confidence = predict_face_shape(face)

        st.success(f"Face Shape: {label}")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

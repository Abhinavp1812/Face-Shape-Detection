import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square"
]

MODEL_PATH = "best_model2.pth"

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    # ✅ use weights=None instead of pretrained=False
    model = torchvision.models.efficientnet_b4(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
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
def predict_face_shape(image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

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

        # Fix mirrored webcam image
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

# -------- Upload input --------
else:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# -------- Prediction --------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Analyzing face shape..."):
        label, confidence = predict_face_shape(image)

    st.success(f"Face Shape: {label}")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

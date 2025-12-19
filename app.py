import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import mediapipe as mp





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

def detect_face(image: Image.Image):
    import mediapipe as mp

    mp_face_detection = mp.solutions.face_detection

    img = np.array(image)
    h, w, _ = img.shape

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    ) as face_detection:

        results = face_detection.process(img)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        pad = int(0.15 * bh)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x1 + bw + pad)
        y2 = min(h, y1 + bh + pad)

        face_crop = img[y1:y2, x1:x2]
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

        with st.spinner("Analyzing face shape......."):
            label, confidence = predict_face_shape(face)

        st.success(f"Face Shape: {label}")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

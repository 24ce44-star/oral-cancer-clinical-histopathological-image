import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import io

# ----------------------------
# CONFIG
# ----------------------------
CLINICAL_RESNET_WEIGHTS = r"best_resnet50_clinical_windows.pth"
HISTO_RESNET_WEIGHTS = r"oral_resnet50_finetuned.pth"

CLINICAL_TEST_FOLDER = r"test_clinical"
HISTO_TEST_FOLDER = r"test_histopath"

# ----------------------------
# LABEL MAPS
# ----------------------------
HISTO_CLASS_NAMES = ["NON CANCER", "CANCER"]       # 0,1
CLINICAL_CLASS_NAMES = ["CANCER", "NON CANCER"]    # 0,1 (trained ulta)

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_clinical_resnet(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_histopath_resnet(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


clinical_model = load_clinical_resnet(CLINICAL_RESNET_WEIGHTS)
histo_model = load_histopath_resnet(HISTO_RESNET_WEIGHTS)

# ----------------------------
# TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# DRAW LABEL ON IMAGE (FIXED)
# ----------------------------
def draw_label_on_image(pil_img, label, confidence):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    text = f"{label} ({confidence*100:.1f}%)"

    # ✅ CORRECT COLOR LOGIC
    if label == "NON CANCER":
        color = (0, 255, 0)   # Green
    else:
        color = (0, 0, 255)   # Red

    cv2.rectangle(img, (10, 10), (450, 60), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------
# PREDICTION HELPERS
# ----------------------------
def predict_image(model_device, pil_img, class_names):
    model, device = model_device
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
    idx = int(torch.argmax(prob))
    return class_names[idx], float(prob[idx])


def predict_folder(model_device, folder, class_names, max_images=500):
    results = []
    for i, f in enumerate(sorted(os.listdir(folder))):
        if i >= max_images:
            break
        if not f.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".bmp")):
            continue
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        label, conf = predict_image(model_device, img, class_names)
        overlay = draw_label_on_image(img, label, conf)
        results.append((f, overlay))
    return results

# ----------------------------
# WEBCAM (CLINICAL)
# ----------------------------
def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Clinical Oral Image | SPACE = capture | Q = quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            captured = buf.getvalue()
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Oral Cancer Detection", layout="wide")
st.title("🦷 Oral Cancer Detection System")

tab1, tab2 = st.tabs(["Clinical Oral Images", "Histopathological Images"])

# ----------------------------
# TAB 1: CLINICAL
# ----------------------------
with tab1:
    option = st.radio("Choose input method", ("Upload Image", "Use Webcam", "Run on Test Folder"))

    if option == "Upload Image":
        up = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if up:
            img = Image.open(up).convert("RGB")
            label, conf = predict_image(clinical_model, img, CLINICAL_CLASS_NAMES)
            st.image(draw_label_on_image(img, label, conf), use_column_width=True)

    elif option == "Use Webcam":
        if st.button("Open Webcam"):
            img_bytes = capture_from_webcam()
            if img_bytes:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                label, conf = predict_image(clinical_model, img, CLINICAL_CLASS_NAMES)
                st.image(draw_label_on_image(img, label, conf), use_column_width=True)

    else:
        if st.button("Run on Clinical Test Folder"):
            res = predict_folder(clinical_model, CLINICAL_TEST_FOLDER, CLINICAL_CLASS_NAMES)
            cols = st.columns(4)
            for i, (f, img) in enumerate(res):
                with cols[i % 4]:
                    st.image(img, use_column_width=True)
                    st.caption(f)

# ----------------------------
# TAB 2: HISTOPATH
# ----------------------------
with tab2:
    option = st.radio("Choose input method", ("Upload Image", "Run on Test Folder"))

    if option == "Upload Image":
        up = st.file_uploader("Upload H&E image", type=["jpg", "png", "jpeg", "tif"])
        if up:
            img = Image.open(up).convert("RGB")
            label, conf = predict_image(histo_model, img, HISTO_CLASS_NAMES)
            st.image(draw_label_on_image(img, label, conf), use_column_width=True)

    else:
        if st.button("Run on Histopathology Test Folder"):
            res = predict_folder(histo_model, HISTO_TEST_FOLDER, HISTO_CLASS_NAMES)
            cols = st.columns(4)
            for i, (f, img) in enumerate(res):
                with cols[i % 4]:
                    st.image(img, use_column_width=True)
                    st.caption(f)

st.caption("🟢 Non-Cancer = Green | 🔴 Cancer = Red")

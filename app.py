import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy
import pywt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# =============================
# MODEL
# =============================

class DeepFakeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        self.classifier = nn.Sequential(

            nn.Linear(1536,512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(512,2)
        )

    def forward(self,x):

        x = self.backbone(x)

        return self.classifier(x)


# =============================
# LOAD MODEL
# =============================

@st.cache_resource
def load_model():

    model = DeepFakeNet()

    state_dict = torch.load(
        "deepfake_detector_model.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# =============================
# IMAGE TRANSFORM
# =============================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# =============================
# FEATURE EXTRACTION
# =============================

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sk = skew(gray.flatten())

    kt = kurtosis(gray.flatten())

    ent = shannon_entropy(gray)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm,'contrast')[0,0]

    homogeneity = graycoprops(glcm,'homogeneity')[0,0]

    gradient = sobel(gray)

    grad_mean = np.mean(gradient)

    coeffs = pywt.dwt2(gray,'haar')

    cA,(cH,cV,cD) = coeffs

    wavelet_energy = np.sum(cH**2 + cV**2 + cD**2)

    return {
        "Skewness": sk,
        "Kurtosis": kt,
        "Entropy": ent,
        "Texture Contrast": contrast,
        "Texture Homogeneity": homogeneity,
        "Gradient Mean": grad_mean,
        "Wavelet Energy": wavelet_energy
    }


# =============================
# STREAMLIT UI
# =============================

st.title("🔍 DeepFake Detection Dashboard")

st.markdown(
"""
Upload an image to analyze whether it is **Real or DeepFake** using a CNN-based model.
"""
)

uploaded_file = st.file_uploader(
"Upload Image",
type=["jpg","jpeg","png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    st.subheader("Uploaded Image")

    st.image(img_np,use_container_width=True)

    # =============================
    # MODEL INPUT
    # =============================

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probs = torch.softmax(output,dim=1)[0]

    fake_prob = probs[0].item()

    real_prob = probs[1].item()

    prediction = "REAL" if real_prob > fake_prob else "FAKE"

    confidence = max(fake_prob,real_prob)


    # =============================
    # RESULT
    # =============================

    st.subheader("Prediction Result")

    if prediction == "REAL":

        st.success("REAL IMAGE")

    else:

        st.error("FAKE IMAGE")

    st.write(f"Confidence: {confidence:.3f}")


    # =============================
    # PROBABILITY CHART
    # =============================

    st.subheader("Prediction Probability")

    labels = ["Fake","Real"]

    values = [fake_prob,real_prob]

    fig,ax = plt.subplots()

    ax.bar(labels,values)

    ax.set_ylabel("Probability")

    st.pyplot(fig)


    # =============================
    # GRADCAM
    # =============================

    st.subheader("GradCAM Visualization")

    target_layer = model.backbone.conv_head

    cam = GradCAM(model=model,target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # FIX: Resize heatmap to image size
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (img_np.shape[1], img_np.shape[0])
    )

    img_float = np.float32(img_np)/255

    cam_image = show_cam_on_image(
        img_float,
        grayscale_cam,
        use_rgb=True
    )

    st.image(cam_image)


    # =============================
    # FEATURES
    # =============================

    features = extract_features(img_np)

    st.subheader("Image Statistical Parameters")

    for k,v in features.items():

        st.write(f"{k}: {v:.4f}")


    # =============================
    # FEATURE VISUALIZATION
    # =============================

    st.subheader("Feature Visualization")

    labels = list(features.keys())

    values = list(features.values())

    fig,ax = plt.subplots(figsize=(8,4))

    ax.bar(labels,values)

    ax.set_xticklabels(labels,rotation=45)

    st.pyplot(fig)


    # =============================
    # EXPLANATION
    # =============================

    st.subheader("Feature Explanation")

    st.markdown("""
**Skewness** – Asymmetry of pixel intensity distribution.

**Kurtosis** – Sharpness of pixel intensity distribution.

**Entropy** – Randomness in the image.

**Texture (GLCM)** – Measures spatial texture patterns.

**Gradient** – Represents edge intensity.

**Wavelet Energy** – Captures high-frequency artifacts common in deepfake images.
""")


# =============================
# SIDEBAR
# =============================

st.sidebar.title("Model Information")

st.sidebar.markdown("""
Architecture: EfficientNet-B3  
Input Size: 224x224  
Classes: Real / Fake  
Framework: PyTorch  

This system detects manipulated facial images generated using deepfake techniques.
""")
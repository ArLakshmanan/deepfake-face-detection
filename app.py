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
# WAVELET MAP
# =============================

def generate_wavelet_map(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(gray,'haar')

    cA,(cH,cV,cD) = coeffs

    hf = np.abs(cH) + np.abs(cV) + np.abs(cD)

    hf = cv2.normalize(hf,None,0,255,cv2.NORM_MINMAX)

    return hf


# =============================
# ENTROPY HISTOGRAM
# =============================

def generate_entropy_histogram(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])

    return hist


# =============================
# GEOMETRY MAP
# =============================

def generate_geometry_map(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,100,200)

    return edges


# =============================
# STREAMLIT UI
# =============================

st.title("DeepFake Detection System")

st.write(
"Upload an image to detect whether it is REAL or FAKE."
)

uploaded_file = st.file_uploader(
"Upload Image",
type=["jpg","jpeg","png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    # =============================
    # MODEL PREDICTION
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
    # GENERATE VISUALS
    # =============================

    wavelet_map = generate_wavelet_map(img_np)

    entropy_hist = generate_entropy_histogram(img_np)

    geometry_map = generate_geometry_map(img_np)


    st.subheader("Analysis Results")

    col1,col2 = st.columns(2)

    with col1:

        st.image(
            img_np,
            caption=f"Original Image\nPrediction: {prediction}\nConfidence: {confidence*100:.2f}%",
            use_container_width=True
        )

        st.image(
            wavelet_map,
            caption="Wavelet High Frequency Map",
            use_container_width=True
        )

    with col2:

        fig,ax = plt.subplots()

        ax.plot(entropy_hist)

        ax.set_title("Entropy Histogram")

        ax.set_xlabel("Pixel Intensity")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        st.image(
            geometry_map,
            caption="Geometry / Edge Map",
            use_container_width=True
        )


    # =============================
    # FEATURE VALUES
    # =============================

    features = extract_features(img_np)

    st.subheader("Extracted Statistical Features")

    for k,v in features.items():

        st.write(f"{k}: {v:.4f}")

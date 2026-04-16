import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image
from skimage.measure import shannon_entropy
import pywt

# -------------------------------
# MODEL
# -------------------------------

class DeepFakeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        self.fc = nn.Linear(1536,2)

    def forward(self,x):

        x = self.backbone(x)

        return self.fc(x)


# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():

    model = DeepFakeNet()

    state = torch.load(
        "deepfake_detector_model.pth",
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


model = load_model()

# -------------------------------
# TRANSFORM
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -------------------------------
# WAVELET MAP
# -------------------------------

def wavelet_map(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(gray,"haar")

    cA,(cH,cV,cD) = coeffs

    hf = np.abs(cH)+np.abs(cV)+np.abs(cD)

    hf = cv2.normalize(hf,None,0,255,cv2.NORM_MINMAX)

    return hf.astype(np.uint8)

# -------------------------------
# EDGE MAP
# -------------------------------

def geometry_map(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,100,200)

    return edges


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("DeepFake Face Detection")

st.write(
"Upload a face image to detect whether it is REAL or FAKE"
)

uploaded_file = st.file_uploader(
"Upload Image",
type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    # prediction
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        out = model(input_tensor)

        prob = torch.softmax(out,dim=1)[0]

    fake_prob = prob[0].item()

    real_prob = prob[1].item()

    prediction = "REAL" if real_prob > fake_prob else "FAKE"

    confidence = max(real_prob,fake_prob)

    # maps
    wave = wavelet_map(img_np)

    edges = geometry_map(img_np)

    entropy = shannon_entropy(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY))

    # display
    col1,col2 = st.columns(2)

    with col1:

        st.image(
            img_np,
            caption=f"Prediction: {prediction} | Confidence: {confidence*100:.2f}%"
        )

        st.image(
            wave,
            caption="Wavelet High Frequency Map"
        )

    with col2:

        fig,ax = plt.subplots()

        hist = cv2.calcHist(
            [cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)],
            [0],
            None,
            [256],
            [0,256]
        )

        ax.plot(hist)

        ax.set_title("Pixel Intensity Histogram")

        st.pyplot(fig)

        st.image(
            edges,
            caption="Geometry / Edge Map"
        )

    st.write(f"Image Entropy: {entropy:.3f}")

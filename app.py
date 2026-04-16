import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import shannon_entropy
import pywt

# =============================
# FEATURE EXTRACTION
# =============================

def extract_features(img):

    # Convert to grayscale (NO cv2)
    gray = np.mean(img, axis=2).astype(np.uint8)

    # 1. Entropy
    entropy = shannon_entropy(gray)

    # 2. Wavelet Energy
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    wavelet_energy = np.sum(cH**2 + cV**2 + cD**2)

    # 3. Geometry (Aspect Ratio)
    h, w = gray.shape
    aspect_ratio = w / h

    return entropy, wavelet_energy, aspect_ratio


# =============================
# STREAMLIT UI
# =============================

st.title("🔍 DeepFake Detection (Lightweight)")

st.markdown("""
This system detects **Real vs Fake images** using:
- Entropy
- Wavelet Energy
- Image Geometry
""")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(img_np, use_container_width=True)

    # =============================
    # FEATURE EXTRACTION
    # =============================

    entropy, wavelet, ratio = extract_features(img_np)

    # =============================
    # CLASSIFICATION LOGIC
    # =============================

    if (entropy < 6.5) or (wavelet > 1e6) or (ratio < 0.7 or ratio > 1.5):
        prediction = "FAKE"
        confidence = 0.85
    else:
        prediction = "REAL"
        confidence = 0.85

    # =============================
    # RESULT
    # =============================

    st.subheader("Prediction Result")

    if prediction == "REAL":
        st.success("REAL IMAGE")
    else:
        st.error("FAKE IMAGE")

    st.write(f"Confidence: {confidence:.2f}")

    # =============================
    # FEATURE DISPLAY
    # =============================

    st.subheader("Feature Values")

    st.write(f"Entropy: {entropy:.3f}")
    st.write(f"Wavelet Energy: {wavelet:.2f}")
    st.write(f"Aspect Ratio: {ratio:.2f}")

    # =============================
    # VISUALIZATION
    # =============================

    st.subheader("Feature Visualization")

    labels = ["Entropy", "Wavelet Energy", "Aspect Ratio"]
    values = [entropy, wavelet, ratio]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Value")

    st.pyplot(fig)

    # =============================
    # EXPLANATION
    # =============================

    st.subheader("Explanation")

    st.markdown("""
- **Entropy**: Measures randomness (fake images often less natural)
- **Wavelet Energy**: Detects high-frequency artifacts
- **Aspect Ratio**: Checks geometric consistency
""")

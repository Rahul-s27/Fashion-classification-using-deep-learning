import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Page config & custom styling
# ---------------------------
st.set_page_config(page_title="Fashion Classifier", layout="centered")

custom_css = """
<style>
body {
    background-color: #0f0f0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #d1d5db;
}
.stButton > button {
    background-color: #1f2937;
    color: #ffffff;
    border-radius: 8px;
    border: none;
}
.stUploadDropzone {
    background-color: #1e293b;
    border: 1px dashed #3b82f6;
}
.stDataFrame, .stTable {
    background-color: #1e293b;
    color: #e5e7eb;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# Load model and label encoder
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

@st.cache_resource
def load_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.load("label_encoder.joblib", allow_pickle=True)
    return le

model = load_model()
label_encoder = load_label_encoder()

# ---------------------------
# Sidebar instructions
# ---------------------------
with st.sidebar:
    st.title("Instructions")
    st.markdown("""
        1. Upload a fashion product image (JPG/PNG).
        2. The model predicts the clothing category.
        3. See the top-5 predictions with confidence.
    """)

# ---------------------------
# Main App
# ---------------------------
st.title("Fashion Image Classifier")
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = Image.open(image).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_confidences = predictions[top_indices]
    top_classes = label_encoder.inverse_transform(top_indices)

    # Top-5 Results Table
    st.subheader("Top-5 Predictions")
    df = pd.DataFrame({
        "Class": top_classes,
        "Confidence (%)": (top_confidences * 100).round(2)
    })
    st.dataframe(df.style.format({'Confidence (%)': '{:.2f}'}).background_gradient(cmap='Blues'))

    # Bar Chart
    st.subheader("Model Confidence")
    fig, ax = plt.subplots()
    sns.barplot(x=top_confidences * 100, y=top_classes, palette="Blues_d")
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

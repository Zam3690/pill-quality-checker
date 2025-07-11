import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from helper import preprocess_image, load_labels

# Load model and labels
model = load_model("keras_model_fixed1.h5")
class_labels = load_labels("labels1.txt")

st.set_page_config(page_title="Pill Quality Checker")
st.title("💊 Pill Defect Detection using AI")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the pill", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)[0]

    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.subheader("🔍 Prediction Result")
    st.success(f"Prediction: **{predicted_label}** ({confidence:.2f}%)")

    if predicted_label != "Good":
        st.warning(f"⚠️ Defect Type Detected: **{predicted_label}**")

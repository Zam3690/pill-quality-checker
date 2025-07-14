import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from helper import preprocess_image, load_labels

# Load model and labels
model = load_model("keras_model_fixed1.h5")
class_labels = load_labels("labels.txt")

st.set_page_config(page_title="Pill Quality Checker")
st.title("💊 Pill Defect Detection (Upload & Webcam)")

# --- Helper function to crop the pill ---
def detect_and_crop_pill(frame, draw_box=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 10
    x, y = max(x - pad, 0), max(y - pad, 0)
    cropped = frame[y:y + h + pad, x:x + w + pad]

    if draw_box:
        cv2.rectangle(frame, (x, y), (x + w + pad, y + h + pad), (0, 255, 0), 2)

    return cropped, frame

# --- Image Prediction ---
st.subheader("📤 Upload Pill Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.success(f"Prediction: **{predicted_label}** ({confidence:.2f}%)")
    if predicted_label != "Good":
        st.warning(f"⚠️ Defect Type Detected: **{predicted_label}**")

st.markdown("---")

# --- Webcam Start/Stop Buttons ---
st.subheader("📷 Real-time Webcam Pill Detection")

if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Webcam"):
        st.session_state.webcam_active = True
with col2:
    if st.button("⏹ Stop Webcam"):
        st.session_state.webcam_active = False

FRAME_WINDOW = st.image([])

if st.session_state.webcam_active:
    camera = cv2.VideoCapture(0)

    while st.session_state.webcam_active:
        success, frame = camera.read()
        if not success:
            st.error("⚠️ Could not access webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_pill, boxed_frame = detect_and_crop_pill(frame, draw_box=True)

        if cropped_pill is not None:
            pill_img = Image.fromarray(cropped_pill)
            input_data = preprocess_image(pill_img)

            prediction = model.predict(input_data)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_index]
            confidence = prediction[predicted_index] * 100

            cv2.putText(
                boxed_frame,
                f"{predicted_label} ({confidence:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
        else:
            cv2.putText(
                boxed_frame,
                "No pill detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        FRAME_WINDOW.image(boxed_frame)

    camera.release()

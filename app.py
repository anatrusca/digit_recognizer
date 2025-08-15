import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🖌️ Handwritten Digit Recognizer")


@st.cache_resource
def load_digit_model():
    return load_model("saved_model/digit_model.h5")


model = load_digit_model()
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    # Convert to grayscale
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert if white background
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    # Normalize and reshape
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        processed_img = preprocess_image(
            canvas_result.image_data.astype(np.uint8)
        )
        if processed_img is not None:
            prediction = model.predict(processed_img)
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            st.success(f"✅ Predicted Digit: **{predicted_digit}**")
            st.info(f"🔢 Confidence: **{confidence:.2f}%**")
            st.write("🔍 Full Prediction Probabilities:")
            fig, ax = plt.subplots()
            ax.bar(range(10), prediction[0])
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            st.pyplot(fig)
        else:
            st.warning("Could not preprocess the input image.")
    else:
        st.warning("Please draw a digit on the canvas before predicting.")

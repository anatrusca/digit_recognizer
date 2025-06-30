import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from keras.models import load_model

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🖌️ Handwritten Digit Recognizer")

model = load_model("saved_model/digit_model.h5")
st.write("Model loaded.")

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

canvas_result = st_canvas(
    fill_color="black",   
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:

    st.image(canvas_result.image_data, caption="🖼️ Your drawing", width=150)

    if st.button("🧠 Predict"):
        img = canvas_result.image_data.astype(np.uint8)
        processed = preprocess_image(img)

        # Show processed image
        st.image(processed.reshape(28, 28), caption="🔍 Processed Image", width=150)

        # Prediction
        prediction = model.predict(processed)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.write(f"### 🎯 Prediction: **{pred_class}**")
        st.write(f"Confidence: `{confidence:.2f}%`")

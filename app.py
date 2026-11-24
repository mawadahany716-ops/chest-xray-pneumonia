import streamlit as st
import numpy as np
import cv2
import joblib

st.set_page_config(page_title="Chest X-Ray Classification")

st.title("Chest X-Ray Pneumonia Classification")
st.write("Upload a chest X-ray image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an X-Ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded X-Ray", width=300)

    img = cv2.resize(img, (32, 32))
    img = img.flatten() / 255.0
    img = img.reshape(1, -1)

   model = joblib.load("best_knn_model_compressed.pkl")

    if st.button("Predict"):

        prediction = model.predict(img)[0]

        classes = {
            0: "Normal",
            1: "Bacterial Pneumonia",
            2: "Viral Pneumonia"
        }

        st.success("Prediction: " + classes[prediction])

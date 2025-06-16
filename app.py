import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Comic Face Swap", layout="centered")
st.title("💥 Comic Book Face Swapper")
st.markdown("Upload a single-female comic cover and your selfie to create a custom version!")

cover_file = st.file_uploader("Upload Comic Book Cover", type=["jpg", "jpeg", "png"])
selfie_file = st.file_uploader("Upload Your Face Image", type=["jpg", "jpeg", "png"])

if cover_file and selfie_file:
    cover = Image.open(cover_file).convert("RGB")
    selfie = Image.open(selfie_file).convert("RGB")
    cover_np = np.array(cover)
    selfie_np = np.array(selfie)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(cv2.cvtColor(cover_np, cv2.COLOR_BGR2GRAY),
                                          scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        st.warning("No face detected in the comic cover.")
    else:
        x, y, w, h = faces[0]
        selfie_resized = cv2.resize(selfie_np, (w, h))
        cover_np[y : y + h, x : x + w] = selfie_resized
        result = Image.fromarray(cover_np)
        st.image(result, caption="Your Custom Comic Cover", use_column_width=True)
        buf = cv2.imencode(".jpg", cover_np)[1].tobytes()
        st.download_button("Download Your Image", data=buf, 
                           file_name="comic_swap.jpg", mime="image/jpeg")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib

st.title("Comic Book Face Swap (Styled, Cloud-Friendly)")

uploaded_comic = st.file_uploader("Upload a comic book cover (JPG/PNG)", type=["jpg", "jpeg", "png"], key="comic")
uploaded_face = st.file_uploader("Upload your face (selfie)", type=["jpg", "jpeg", "png"], key="face")

detector = dlib.get_frontal_face_detector()

def detect_face_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if faces:
        face = faces[0]
        return face.left(), face.top(), face.width(), face.height()
    else:
        return None

def apply_comic_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def resize_to_match(source, target):
    return cv2.resize(source, (target.shape[1], target.shape[0]))

if uploaded_comic and uploaded_face:
    comic_img = Image.open(uploaded_comic).convert('RGB')
    face_img = Image.open(uploaded_face).convert('RGB')

    comic_np = np.array(comic_img)
    face_np = np.array(face_img)

    comic_face = detect_face_dlib(comic_np)
    user_face = detect_face_dlib(face_np)

    if not comic_face:
        st.error("No face detected on comic cover.")
    elif not user_face:
        st.error("No face detected in selfie.")
    else:
        x_c, y_c, w_c, h_c = comic_face
        x_u, y_u, w_u, h_u = user_face

        face_crop = face_np[y_u:y_u+h_u, x_u:x_u+w_u]
        face_cartoon = apply_comic_filter(face_crop)
        face_resized = resize_to_match(face_cartoon, comic_np[y_c:y_c+h_c, x_c:x_c+w_c])

        result_img = comic_np.copy()
        result_img[y_c:y_c+h_c, x_c:x_c+w_c] = face_resized

        st.image(result_img, caption="Face-swapped Comic", use_column_width=True)
        result_pil = Image.fromarray(result_img)
        st.download_button("Download Your Comic Cover", data=result_pil.tobytes(), file_name="comic_face_swap.png", mime="image/png")

else:
    st.info("Please upload both a comic book cover and your selfie to continue.")

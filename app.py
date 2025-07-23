import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.title("🦸 Comic Book Face Swap (OpenCV Edition)")

uploaded_comic = st.file_uploader("📕 Upload a comic book cover (JPG/PNG)", type=["jpg", "jpeg", "png"], key="comic")
uploaded_face = st.file_uploader("🤳 Upload your selfie", type=["jpg", "jpeg", "png"], key="face")

# Load face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # Return the largest face detected
        biggest_face = max(faces, key=lambda r: r[2]*r[3])
        return biggest_face
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

    comic_face = detect_face_opencv(comic_np)
    user_face = detect_face_opencv(face_np)

    if not comic_face:
        st.error("😢 No face detected on the comic book cover.")
    elif not user_face:
        st.error("😢 No face detected in your selfie.")
    else:
        x_c, y_c, w_c, h_c = comic_face
        x_u, y_u, w_u, h_u = user_face

        face_crop = face_np[y_u:y_u+h_u, x_u:x_u+w_u]
        face_cartoon = apply_comic_filter(face_crop)
        face_resized = resize_to_match(face_cartoon, comic_np[y_c:y_c+h_c, x_c:x_c+w_c])

        result_img = comic_np.copy()
        result_img[y_c:y_c+h_c, x_c:x_c+w_c] = face_resized

        st.image(result_img, caption="✅ Face-swapped Comic", use_column_width=True)

        # Debug previews
        with st.expander("🔍 Debug: Preview Cropped Faces"):
            st.image(face_crop, caption="Cropped Selfie Face", channels="RGB")
            st.image(face_resized, caption="Cartoonized + Resized Face", channels="RGB")

        # Convert to BytesIO for download
        result_pil = Image.fromarray(result_img)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("⬇️ Download Your Comic Cover", data=byte_im, file_name="comic_face_swap.png", mime="image/png")

else:
    st.info("📂 Please upload both a comic book cover and your selfie to get started.")

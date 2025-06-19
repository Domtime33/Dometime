import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Comic Book Face Swap (Styled)")

uploaded_comic = st.file_uploader("Upload a comic book cover (JPG/PNG)", type=["jpg", "jpeg", "png"], key="comic")
uploaded_face = st.file_uploader("Upload your face (selfie)", type=["jpg", "jpeg", "png"], key="face")

def detect_face_mediapipe(image):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if results.detections:
            box = results.detections[0].location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w_box, h_box = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
            return x, y, w_box, h_box
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

    comic_face = detect_face_mediapipe(comic_np)
    user_face = detect_face_mediapipe(face_np)

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

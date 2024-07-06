import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import requests
import logging
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fungsi untuk mengunduh file jika belum ada atau unduhan sebelumnya tidak lengkap
def download_file(url, output_path, expected_size=None):
    if not os.path.exists(output_path) or (expected_size and os.path.getsize(output_path) < expected_size):
        logger.info(f"Downloading {url} to {output_path}...")
        try:
            response = requests.get(url, stream=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            st.error(f"Error downloading {url}: {e}")

# Mengunduh model YOLOv5
@st.cache_resource
def load_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Definisikan subset label yang diizinkan
allowed_labels = {"person", "car", "motorbike", "bus", "truck", "train", "bicycle", "traffic light", "parking meter", "stop sign"} 

# Fungsi untuk deteksi objek
def detect_objects(model, image, allowed_labels):
    results = model(image)
    labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    height, width, _ = image.shape

    for i in range(len(labels)):
        label_name = model.names[int(labels[i])]
        if label_name in allowed_labels:
            x1, y1, x2, y2 = int(coords[i][0] * width), int(coords[i][1] * height), int(coords[i][2] * width), int(coords[i][3] * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label_name} {coords[i][-1]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            logger.info(f"Detected label '{label_name}' is not in allowed labels")

    return image

# Fungsi untuk deteksi objek di video
def detect_video(model, video_path, allowed_labels):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'output.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame = detect_objects(model, frame, allowed_labels)
        out.write(detected_frame)

    cap.release()
    out.release()
    return output_video_path

# VideoTransformerBase subclass for real-time object detection
class YOLOv5VideoTransformer(VideoTransformerBase):
    def __init__(self, model, allowed_labels):
        self.model = model
        self.allowed_labels = allowed_labels

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        detected_image = detect_objects(self.model, image, self.allowed_labels)
        return detected_image

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Object Detection using YOLOv5")
    st.write("Upload an image, video, or use your webcam for object detection")

    model = load_yolo()

    option = st.selectbox('Choose an option:', ('Image', 'Video', 'Webcam'))

    if option == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)

            if st.button("Detect Objects"):
                st.write("Detecting...")
                detected_image = detect_objects(model, image, allowed_labels)
                st.image(detected_image, channels="BGR", caption='Detected Image.', use_column_width=True)

                # Opsi unduh gambar hasil deteksi
                is_success, buffer = cv2.imencode(".jpg", detected_image)
                if is_success:
                    st.download_button(
                        label="Download Detected Image",
                        data=buffer.tobytes(),
                        file_name="detected_image.jpg",
                        mime="image/jpeg"
                    )

                # Opsi kembali ke tampilan awal
                if st.button("Back to Start"):
                    st.experimental_rerun()

    elif option == 'Video':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if st.button("Detect Objects in Video"):
                st.write("Detecting...")
                output_video_path = detect_video(model, video_path, allowed_labels)
                st.video(output_video_path)
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Detected Video",
                        data=file,
                        file_name="output_detection.mp4",
                        mime="video/mp4"
                    )

                # Opsi kembali ke tampilan awal
                if st.button("Back to Start"):
                    st.experimental_rerun()

    elif option == 'Webcam':
        camera_option = st.selectbox('Select Camera:', ('Front Camera', 'Back Camera'))
        if camera_option == 'Front Camera':
            selected_device = {'label': 'Front Camera', 'id': 'user'}
        else:
            selected_device = {'label': 'Back Camera', 'id': 'environment'}

        webrtc_streamer(
            key="example",
            video_transformer_factory=lambda: YOLOv5VideoTransformer(model, allowed_labels),
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={
                "video": {
                    "facingMode": {"exact": selected_device['id']}
                },
                "audio": False
            },
            async_transform=True
        )

        if st.button("Back to Start"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()

st.caption('Copyright (C) Shafira Fimelita Q - 2024')

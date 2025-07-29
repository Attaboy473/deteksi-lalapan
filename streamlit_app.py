import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import time
import threading

# Clear the cache to ensure the model is reloaded with the correct device bagian 
st.cache_resource.clear()

# --- Configuration (from opencv_tester.py) ---
TH_LOW = 0.4
TH_UNKNOWN = 0.6
TH_MID = 0.85
IMG_SIZE = 224

# --- Preprocessing Transform (from opencv_tester.py) ---
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        model = YOLO(model_path).to("cpu")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Classification Logic (adapted from opencv_tester.py) ---
def classify_image_with_model(frame, model):
    """
    Classify the frame using the pre-trained YOLO model with custom thresholds:
    <0.6 unknown, 0.6-0.85 'Agak Layu' for wilted, >=0.85 humanized label
    """
    results = model(frame)  # Perform inference (YOLOv8 API is compatible)
    predictions = results[0]
    if hasattr(predictions, 'boxes') and len(predictions.boxes) > 0:
        class_ids = predictions.boxes.cls.cpu().numpy()
        confidences = predictions.boxes.conf.cpu().numpy()
        conf = confidences[0]
        class_names = ['SELADA_LAYU', 'SELADA_SEGAR', 'TIMUN_LAYU', 'TIMUN_SEGAR']
        raw_label = class_names[int(class_ids[0])]
        if conf < 0.6:
            return 'UNKNOWN', conf
        if conf < 0.85 and raw_label in ['SELADA_LAYU', 'TIMUN_LAYU']:
            return ('Selada Agak Layu' if raw_label == 'SELADA_LAYU' else 'Timun Agak Layu'), conf
        label = raw_label.replace('_', ' ').title()
        return label, conf
    return 'UNKNOWN', 0

# --- UI ---
st.title("Klasifikasi Kesegaran Sayuran")
st.write("Aplikasi untuk mendeteksi kesegaran Selada dan Timun menggunakan YOLO.")

# --- Configuration ---
# You can change the model path here if needed.
MODEL_PATH = "attamodel.pt"
model = load_yolo_model(MODEL_PATH)

if model:
    # Sidebar for input method selection
    input_method = st.sidebar.selectbox(
        "Pilih Metode Input:",
        ("Unggah Gambar", "Ambil Foto via Kamera", "Kamera Realtime")
    )

    # --- Unggah Gambar ---
    if input_method == "Unggah Gambar":
        st.header("Unggah Gambar")
        uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)
            
            # Convert PIL image to OpenCV format
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            with st.spinner("Menganalisis gambar..."):
                label, confidence = classify_image_with_model(frame, model)
            
            st.success(f"Hasil Klasifikasi: **{label}**")
            st.write(f"Tingkat Keyakinan: **{confidence:.2f}**")

    # --- Ambil Foto via Kamera ---
    elif input_method == "Ambil Foto via Kamera":
        st.header("Ambil Foto via Kamera")
        picture = st.camera_input("Arahkan kamera dan ambil gambar")

        if picture:
            image = Image.open(picture)
            st.image(image, caption="Gambar yang diambil", use_column_width=True)

            # Convert PIL image to OpenCV format
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with st.spinner("Menganalisis gambar..."):
                label, confidence = classify_image_with_model(frame, model)

            st.success(f"Hasil Klasifikasi: **{label}**")
            st.write(f"Tingkat Keyakinan: **{confidence:.2f}**")

    # --- Kamera Realtime (OpenCV) ---
    elif input_method == "Kamera Realtime":
        st.header("Kamera Realtime dengan OpenCV")
        run = st.checkbox('Aktifkan Kamera Realtime')
        FRAME_WINDOW = st.image([])
        result_placeholder = st.empty()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Kamera tidak terdeteksi atau tidak bisa diakses. Pastikan webcam terhubung dan tidak digunakan aplikasi lain.")
        else:
            last_pred_time = time.time() - 3
            while run:
                ret, frame = camera.read()
                if not ret or frame is None:
                    st.warning("Gagal membaca frame dari kamera. Coba restart aplikasi atau periksa koneksi webcam.")
                    FRAME_WINDOW.empty()
                    time.sleep(1)
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                FRAME_WINDOW.image(img)
                current_time = time.time()
                if current_time - last_pred_time >= 3:
                    last_pred_time = current_time
                    # Klasifikasi frame setiap 3 detik
                    label, confidence = classify_image_with_model(frame, model)
                    with result_placeholder.container():
                        st.write(f"Label: {label}")
                        st.write(f"Confidence: {confidence * 100:.2f}%")
            camera.release()
            st.write('Kamera dihentikan')

else:
    st.error(f"Model tidak dapat dimuat dari path: `{MODEL_PATH}`. Pastikan file model ada di path yang benar.")

st.sidebar.info(
    "**Thresholds:**\n"
    "- **< 0.6:** Unknown\n"
    "- **0.6 - 0.85:** Agak Layu\n"
    "- **>= 0.85:** Segar / Layu (Full)"
)

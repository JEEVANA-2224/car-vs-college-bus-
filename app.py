import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
class_names = model.names

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Campus Vehicle System", layout="wide")

st.title("🏫 Smart Campus Vehicle Monitoring System 🚗🚍")

# ----------------------------
# SIDEBAR MENU
# ----------------------------
menu = st.sidebar.radio(
    "📌 Select Mode",
    ["Single Image", "Batch Images", "CCTV / Video"]
)

# =========================================================
# 🔹 SINGLE IMAGE MODE
# =========================================================
if menu == "Single Image":

    st.subheader("📸 Single Image Detection")

    file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if file:

        image = Image.open(file)
        img = np.array(image)

        results = model(img)

        annotated = results[0].plot()

        car_count = 0
        bus_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = class_names[cls]

                if label == "car":
                    car_count += 1
                elif label == "bus":
                    bus_count += 1

        st.image(img, caption="Input Image", use_container_width=True)
        st.image(annotated, caption="Detection Result", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🚗 Cars", car_count)
        col2.metric("🚍 Buses", bus_count)
        col3.metric("🚦 Total", car_count + bus_count)

# =========================================================
# 🔹 BATCH MODE
# =========================================================
elif menu == "Batch Images":

    st.subheader("📂 Multiple Image Detection")

    files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if files:

        total_car = 0
        total_bus = 0

        for file in files:

            image = Image.open(file)
            img = np.array(image)

            results = model(img)
            annotated = results[0].plot()

            car_count = 0
            bus_count = 0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = class_names[cls]

                    if label == "car":
                        car_count += 1
                        total_car += 1
                    elif label == "bus":
                        bus_count += 1
                        total_bus += 1

            st.image(annotated, caption=file.name)

        st.markdown("## 📊 Batch Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("🚗 Total Cars", total_car)
        col2.metric("🚍 Total Buses", total_bus)
        col3.metric("🚦 Grand Total", total_car + total_bus)

# =========================================================
# 🔹 CCTV / VIDEO MODE
# =========================================================
elif menu == "CCTV / Video":

    st.subheader("🎥 CCTV / Video Detection")

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:

        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        stframe = st.empty()

        car_count = 0
        bus_count = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            # reset per frame counting (optional live view)
            car_count_frame = 0
            bus_count_frame = 0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = class_names[cls]

                    if label == "car":
                        car_count_frame += 1
                    elif label == "bus":
                        bus_count_frame += 1

            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()

        st.success("✔ Video Processing Completed")
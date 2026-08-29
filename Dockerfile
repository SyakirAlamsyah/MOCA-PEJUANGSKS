FROM ultralytics/ultralytics:latest-jetson-jetpack4
RUN pip3 install numpy==1.23.5 streamlit face_recognition
WORKDIR /workspace
FROM ultralytics/ultralytics:latest-jetson-jetpack4
RUN apt-get update && apt-get install -y build-essential cmake
RUN pip3 install numpy==1.23.5 streamlit face_recognition
WORKDIR /workspace
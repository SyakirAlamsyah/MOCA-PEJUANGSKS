from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.engine", task="detect")
print(model.names)
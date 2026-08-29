from ultralytics import YOLO
model = YOLO("models/best.engine", task="detect")
print(model.names)
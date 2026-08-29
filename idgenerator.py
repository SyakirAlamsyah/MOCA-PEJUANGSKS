from ultralytics import YOLO

opsi = input("Pilih model apa? ")

model = YOLO("models/"+opsi+".engine", task="detect")
print(model.names)

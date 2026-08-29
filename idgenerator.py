from ultralytics import YOLO

model = ""
opsi = int(input("Pilih model apa?(1/2/3/4)"))

if opsi == 1:
    model == "model"
elif opsi == 2:
    model == "model1"
elif opsi == 3:
    model == "best"
elif opsi == 4:
    model == "model3"
else:
    print(f"{opsi} tidak valid")

models = YOLO(f"models/{model}.engine", task="detect")
print(models.names)
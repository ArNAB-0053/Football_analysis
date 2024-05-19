from ultralytics import YOLO
import torch 

print(torch.cuda.is_available())

model = YOLO('models/best.pt')

result = model.predict('vdos/08fd33_4.mp4', save = True, device = 0)

print(result[0])

print("-------------------------------------------------------------")
for box in result[0].boxes:
    print(box)
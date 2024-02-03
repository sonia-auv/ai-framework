from ultralytics import YOLO

model = YOLO('./runs/detect/train7/weights/epoch3.pt')
model.export(format='onnx')

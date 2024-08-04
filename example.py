from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='datasets/robosub_24')




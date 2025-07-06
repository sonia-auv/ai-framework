from ultralytics import YOLO


model_path = 'models/robosub_most_recent.pt'  # path to your custom trained model
# Load a model
model = YOLO(model_path)  # load a custom trained model

# Export the model
model.export(format="onnx")
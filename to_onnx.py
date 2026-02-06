from ultralytics import YOLO


model_path = "/home/raph/Documents/ai-framework/models/robosub-2025-v0.pt"  # path to your custom trained model
# Load a model
model = YOLO(model_path)  # load a custom trained model

# Export the model
model.export(format="onnx")

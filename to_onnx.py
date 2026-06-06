from ultralytics import YOLO


model_path = '/home/sonia/ai/ai-framework/runs/detect/aquadome_v1_2i2/model.pt'  # path to your custom trained model
# Load a model
model = YOLO(model_path)  # load a custom trained model

# Export the model
model.export(format="onnx")
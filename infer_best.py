import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd


model_dir = "./runs/detect/train7/weights"
models = []
models_file_path = os.listdir(model_dir)
models_file_path.sort()
for file in models_file_path:
    models.append(YOLO(os.path.join(model_dir, file)))
image_dir = "./infer"
images = []
images_file_path = os.listdir(image_dir)
images_file_path.sort()
images_file_path.remove(".gitignore")
for file in images_file_path:
    image = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_UNCHANGED)
    width = image.shape[1]
    height = image.shape[0]
    x_offset = (640 - width) // 2
    y_offset = (640 - height) // 2
    expanded_image = np.zeros((640, 640, 3), dtype=np.uint8)
    expanded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    images.append(expanded_image)
table_results = np.zeros((len(models), len(images)), dtype=np.uint8)
i = 0
for model in models:
    results = model(images, conf=0.05)
    j = 0
    for result in results:
        table_results[i, j] = result.boxes.cls.cpu().numpy().size
        j += 1
    i += 1
table_results = table_results.transpose()
df = pd.DataFrame(table_results, columns=models_file_path, index=images_file_path)
print(df.to_string())

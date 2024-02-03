from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("./runs/detect/train7/weights/epoch3.pt")  # pretrained YOLOv8n model
file = cv2.imread("./infer/image1.png", cv2.IMREAD_UNCHANGED)
width = file.shape[1]
height = file.shape[0]
# if width / 640 > height / 640:
#     width, height = 640, int(height * 640 / width)
# else:
#     height, width = 640, int(width * 640 / height)
# file = cv2.resize(file, (width, height), interpolation=cv2.INTER_AREA)
x_offset = (640 - width) // 2
y_offset = (640 - height) // 2
expanded_image = np.zeros((640, 640, 3), dtype=np.uint8)
expanded_image[y_offset:y_offset + height, x_offset:x_offset + width] = file
# Run batched inference on a list of images
results = model([expanded_image], conf=0.05)  # return a list of Results objects
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

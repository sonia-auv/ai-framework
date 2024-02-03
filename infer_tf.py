import tensorflow as tf
import cv2
import numpy as np


yolo_classes = [
    "Bins_Abydos_1", "Bins_Abydos_2", "Bins_Earth_1", "Bins_Earth_2", "Gate_Abydos", "Gate_Earth", "Glyph_Abydos_1",
    "Glyph_Abydos_2", "Glyph_Earth_1", "Glyph_Earth_2", "Stargate_Closed", "Stargate_Open"
]


def parse_row(row):
    xc, yc, w, h = row[:4]
    x1 = (xc-w/2)
    y1 = (yc-h/2)
    x2 = (xc+w/2)
    y2 = (yc+h/2)
    prob = row[4:].max(initial=0)
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1, y1, x2, y2, label, prob]


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2-x1)*(y2-y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2)/union(box1, box2)


loaded_model = tf.saved_model.load("./test")
inference = loaded_model.signatures["serving_default"]
img = cv2.imread("./infer/image1.png", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sq_img = np.zeros((640, 640, 3), dtype=np.uint8)
sq_img[120:520, 20:620, :] = img
sq_img = np.transpose(sq_img, (2, 0, 1))
sq_img = np.expand_dims(sq_img, axis=0)
sq_img = sq_img / 255.0
tensor = tf.convert_to_tensor(sq_img.astype(np.float32))

output = inference(images=tensor)["output0"][0].numpy()
output = output.transpose()
boxes = [row for row in [parse_row(row) for row in output] if row[5] > 0.05]
boxes.sort(key=lambda x: x[5], reverse=True)
result = []
while len(boxes) > 0:
    result.append(boxes[0])
    boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]
print(len(result))
print(result)

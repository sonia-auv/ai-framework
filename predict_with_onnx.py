import onnxruntime as ort
import numpy as np
from PIL import Image

CLASSES = {
    0: "gate-sawfish",
    1: "gate-shark",
    2: "gate",
    3: "torpedo-poster",
    4: "torpedo-target",
    5: "sawfish",
    6: "shark",
    7: "bin",
    8: "bin-inner",
    9: "table",
    10: "path",
    11: "red-slalom",
    12: "white-slalom",
    13: "yellow-box",
    14: "pink-box",
    15: "spoon",
    16: "bottle",
}

ratio = 1

# Load the ONNX model
onnx_model_path = (
    "models/model-test-1.onnx"  # Replace with your exported ONNX model path
)
session = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load and preprocess an image for inference
image_path = "/home/raph/Documents/ai-framework/datasets/bottom-maude-et-nimai-lite_nimai-zed_nimai-all_1/test/images/cmczp98yhdh0c0777n7ro9g1o.jpg"  # Replace with your image path
img = Image.open(image_path).resize((608, 608))  # Resize to match model's input size
img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to 0-1
img_np = np.transpose(img_np, (2, 0, 1))[
    np.newaxis, :, :, :
]  # Add batch dimension and transpose

# # Run inference
output = session.run([output_name], {input_name: img_np})


def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy


predict = output[0].squeeze(0).T
predict = predict[predict[:, 4] > 0.1, :]
scores = predict[:, 4]
boxes = predict[:, 0:4] / ratio
boxes = xywh2xyxy(boxes)
kpts = predict[:, 5:]
print(kpts)
for i in range(kpts.shape[0]):
    for j in range(kpts.shape[1] // 3):
        if kpts[i, 3 * j + 2] < self.kpt_score:
            kpts[i, 3 * j : 3 * (j + 1)] = [-1, -1, -1]
        else:
            kpts[i, 3 * j] /= ratio
            kpts[i, 3 * j + 1] /= ratio


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    box and boxes are format as [x1, y1, x2, y2]
    """
    # inter area
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area


def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx + 1]
    return keep_idx


idxes = nms_process(boxes, scores, self.nms_thr)
result = {
    "boxes": boxes[idxes, :].astype(int).tolist(),
    "kpts": kpts[idxes, :].astype(float).tolist(),
    "scores": scores[idxes].tolist(),
}


# from onnxruntime import InferenceSession
# from PIL import Image
# from opyv8 import Predictor
# import cv2
# from time import time

# CLASSES = {
#   0: "gate-sawfish",
#   1: "gate-shark",
#   2: "gate",
#   3: "torpedo-poster",
#   4: "torpedo-target",
#   5: "sawfish",
#   6: "shark",
#   7: "bin",
#   8: "bin-inner",
#   9: "table",
#   10: "path",
#   11: "red-slalom",
#   12: "white-slalom",
#   13: "yellow-box",
#   14: "pink-box",
#   15: "spoon",
#   16: "bottle"
# }

# model = 'models/model-test-1.onnx'
# # List of classes where the index match the class id in the ONNX network
# classes = CLASSES

# IMG_DIR = "/home/raph/Documents/ai-framework/datasets/bottom-maude-et-nimai-lite_nimai-zed_nimai-all_1/test/images/"

# session = InferenceSession(
#     'models/model-test-1.onnx',
# )

# img_name = "cmcs3wbfl92xk0783mj3ub6xf" # "cmczp98yhdh0c0777n7ro9g1o"
# predictor = Predictor(session, list(classes.values()))
# img = Image.open(IMG_DIR+img_name+".jpg")
# output = predictor.predict(img)

# cvimg = cv2.imread(IMG_DIR+img_name+".jpg")

# for label in output.labels:
#     top_left_x = label.x
#     top_left_y  = label.y
#     top_right_x = label.x + int(label.width)
#     top_right_y = label.y+ int(label.height)
#     bottom_right_x = label.x + int(label.width)
#     bottom_right_y = label.y + int(label.height)
#     bottom_left_x = label.x
#     bottom_left_y = label.y


#     cv2.putText(cvimg,
#                 label.classifier,
#                 (int((top_left_x+5)),
#                 int((bottom_right_y-10)/2)),
#                 cv2.FONT_HERSHEY_PLAIN,
#                 .7, (0,0,255), 1, 1)
#     cv2.putText(cvimg,
#                 "{:.1f}%".format(1),
#                 (int((top_left_x+5)),
#                 int((bottom_right_y+10)/2)),
#                 cv2.FONT_HERSHEY_PLAIN,
#                 .7, (0,0,255), 1, 1)
#     cv2.rectangle(cvimg,
#                 (int(top_left_x),int(top_left_y)),
#                 (int(bottom_right_x),int(bottom_right_y)), 
#                 (0,0,255), 1)
# cv2.imwrite('pred_'+str(int(1000*time()))+'.jpg', 
#             cvimg) 
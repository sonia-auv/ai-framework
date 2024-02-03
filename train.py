from ultralytics.yolo.data.converter import convert_coco
import shutil
import json
from ultralytics import YOLO

# Train the model
if __name__ == "__main__":
    convert_coco(labels_dir='datasets/', cls91to80=False)
    with open("datasets/train.json", "r") as f:
        coco_json = json.load(f)
    yaml_content = """path: ./
train: images/train
val: images/val
names:
"""
    for subclass in coco_json["categories"]:
        yaml_content += f"  {subclass['id']-1}: {subclass['name']}\n"
    with open("./yolo_labels/train.yaml", "w") as f:
        f.write(yaml_content)

    shutil.rmtree("./datasets/labels", ignore_errors=True)
    shutil.move("./yolo_labels/labels", "./datasets/labels")

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data='./yolo_labels/train.yaml', epochs=5, imgsz=640, save_period=1, batch=-1, plots=True)

import shutil
import json
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
import torch
import gc


# Train the model
if __name__ == "__main__":
#     convert_coco(labels_dir='datasets/', cls91to80=False)
#     with open("datasets/train.json", "r") as f:
#         coco_json = json.load(f)
#     yaml_content = """path: ./
# train: images/train
# val: images/val
# names:
# """
#     for subclass in coco_json["categories"]:
#         yaml_content += f"  {subclass['id']-1}: {subclass['name']}\n"
#     with open("./yolo_labels/train.yaml", "w") as f:
#         f.write(yaml_content)

#     shutil.rmtree("./datasets/labels", ignore_errors=True)
#     shutil.move("./yolo_labels/labels", "./datasets/labels")
#     model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # Load a model
    # model.train(data='./yolo_labels/train.yaml', epochs=5, imgsz=640, save_period=1, batch=-1, plots=True)
    
    
    
    if torch.cuda.is_available():
        print('Using' ,torch.cuda.get_device_name(torch.cuda.current_device()))
        model = YOLO('yolov8n.pt')
        try:
            results = model.train(data='datasets/coco.yaml', epochs=10, imgsz=640, save_period=1, batch=60, plots=True)
        except:
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print('cuda indisponible, ALLUMES LE GPU !!!')

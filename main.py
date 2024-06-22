from ultralytics import YOLO, YOLOv10
import torch
import argparse
import yaml
import cv2
from time import time
from os import listdir
from os.path import isfile, join

CONFIG_DIR = 'config/'
MODELS_DIR = 'models/'
DATASET_DIR = 'datasets/'
PATH_IMG_TEST = '/test/images'

PARAMETERS_YAML = CONFIG_DIR+'training_parameters.yaml'
AUGMENT_YAML = CONFIG_DIR+'augmentation.yaml'
DEFAULT_YAML = CONFIG_DIR+'.default/ultralytics_default.yaml'
TEMP_YAML = CONFIG_DIR+'.default/ultralytics_custom_config.yaml'
DATASET_YAML = 'data.yaml'


class AiSonia():
    def __init__(self):
        self.parse()

        # Detecting GPU
        if torch.cuda.is_available():
            print('Cuda disponible')
        else:
            print('Cuda non disponible')

        # Loading model
        if self.args.model in ['yolov8', 'yolov9']:
            self.model = YOLO(self.args.load_model)
        elif self.args.model == 'yolov10':
            self.model = YOLOv10(self.args.load_model)

        if self.args.name is None:
            self.args.name = self.args.model

        self.generate_config()

        if self.args.inference:
            self.predict()
        else:
            self.train()


    def parse(self):
        parser = argparse.ArgumentParser(description="Entraînement AI SONIA Vision")
        # Choix du modèle
        parser.add_argument('--project', 
                            type=str, 
                            default=None, 
                            help='Nom du projet (défaut:None)')
        parser.add_argument('--model', 
                            type=str, 
                            required=True, 
                            choices=['yolov8', 'yolov9', 'yolov10'], 
                            help='Modèle choisi (défaut:yolov8)')
        parser.add_argument('--name', 
                            type=str, 
                            default=None, 
                            help='Nom du modèle (défaut:modèle choisi)')
        parser.add_argument('--load-model', 
                            type=str, 
                            required=True, 
                            default=None, 
                            help='Chemin du modèle à charger')
        parser.add_argument('--dataset', 
                            type=str, 
                            default=None, 
                            required=True, 
                            help='Chemin du fichier de configuration du dataset')
        # Entraînement
        parser.add_argument('--resume', 
                            action='store_true', 
                            default=False, 
                            help='Poursuit un entraînement déjà commencé du modèle (défaut:False)')
        # Augmentation des données
        parser.add_argument('--augment', 
                            action='store_true', 
                            default=False, 
                            help='Augmentation des données et entraînement sur ces données augmentées (défaut:False)')
        # Inférence
        parser.add_argument('--inference', 
                            action='store_true', 
                            default=False, 
                            help='Chemin de l\'image sur laquelle une inférence est réalisée (défaut:False)')
        parser.add_argument('--inf-confidence', 
                            type=float, 
                            default=0.5, 
                            help='Confiance minimum pour l\'inférence (défaut:0.5)')
        parser.add_argument('--inf-img-size', 
                            type=int, 
                            default=640, 
                            help='Taille de l\'image pour l\'inférence (défaut:640)')
        # Parsing des arguments
        self.args = parser.parse_args(namespace=None)

    def generate_config(self):
        # Loading configuration
        parameters = yaml.full_load(open(PARAMETERS_YAML))
        settings = yaml.full_load(open(DEFAULT_YAML))
        for key, value in parameters.items():
            settings[key] = value
        
        # Loading augmentation settings
        if self.args.augment:
            augment = yaml.full_load(open(AUGMENT_YAML))
            for key, value in augment.items():
                settings[key] = value
        yaml.dump(settings, open(TEMP_YAML, 'w'))

    def predict(self):
        test_path = DATASET_DIR+self.args.dataset+PATH_IMG_TEST
        img_list = [join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))]
        t1 = time()
        results = self.model.predict(img_list, 
                                     imgsz=self.args.inf_img_size, 
                                     conf=self.args.inf_confidence, 
                                     verbose=False)
        t2 = time()
        print(1/((t2-t1)/len(img_list)), 'fps')
        for i, result in enumerate(results):
            img = cv2.imread(img_list[i])
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                confidence = float(result.boxes.conf[i].item())*100
                bounding_box = result.boxes.xyxy[i].cpu().numpy()
                cv2.putText(img, 
                            name+' '+"{:.1f}%".format(confidence), 
                            (int(bounding_box[0])+10,int(bounding_box[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,0,255), 2, 1)
                cv2.rectangle(img, 
                            (int(bounding_box[0]),int(bounding_box[1])), 
                            (int(bounding_box[2]),int(bounding_box[3])), 
                            (0,0,255), 2)
            cv2.imshow('frame', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def train(self):
        self.model.train(project=self.args.project,
                         name=self.args.name,
                         data=DATASET_DIR+self.args.dataset+'/'+DATASET_YAML,
                         resume=self.args.resume,
                         cfg=TEMP_YAML)
        
def main():
    sonia_ai = AiSonia()

if __name__ == "__main__":
   main()
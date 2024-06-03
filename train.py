from ultralytics import YOLO
import torch
import argparse
import yaml

CONFIG_DIR = 'config/'
MODELS_DIR = 'models/'
DATASET_DIR = 'datasets/'
MODEL_EXTENSION = '.pt'

PARAMETERS_YAML = 'parameters.yaml'
AUGMENT_YAML = 'augment.yaml'
DEFAULT_YAML = CONFIG_DIR+'default.yaml'
TEMP_YAML = CONFIG_DIR+'config.yaml'
DATASET_YAML = 'data.yaml'

def main():
    parser = argparse.ArgumentParser(description="Entraînement AI SONIA Vision")
    # Choix du modèle
    parser.add_argument('--project', type=str, default=None, help='Nom du projet')
    parser.add_argument('--model-name', type=str, default='yolov8n', help='Nom du modèle')
    parser.add_argument('--load-model', type=str, default=None, help='Chemin du modèle à charger')
    parser.add_argument('--dataset', type=str, default=None, help='Chemin du fichier de configuration du dataset')

    # Entraînement
    parser.add_argument('--resume', action='store_true', default=False, help='Poursuit un entraînement déjà commencé du modèle (défaut:False)')

    # Augmentation des données
    parser.add_argument('--augment', action='store_true', default=False, help='Augmentation des données et entraînement sur ces données augmentées')

    # Parsing
    args = parser.parse_args(namespace=None)

    # Detecting GPU
    if torch.cuda.is_available():
        print('Cuda disponible')
    else:
        print('Cuda non disponible')

    # Loading model
    if args.load_model is not None:
        model = YOLO(MODELS_DIR+args.load_model)
    else:
        model = YOLO(args.model_name+MODEL_EXTENSION)

    # Loading configuration
    parameters = yaml.full_load(open(PARAMETERS_YAML))
    settings = yaml.full_load(open(DEFAULT_YAML))
    for key, value in parameters.items():
        settings[key] = value
    
    # Loading augmentation settings
    if args.augment:
        augment = yaml.full_load(open(AUGMENT_YAML))
        for key, value in augment.items():
            settings[key] = value
    yaml.dump(settings, open(TEMP_YAML, 'w'))

    # Training
    model.train(project=args.project,
                name=args.model_name,
                data=DATASET_DIR+args.dataset+'/'+DATASET_YAML, 
                resume=args.resume,
                cfg=TEMP_YAML)

if __name__ == "__main__":
   main()
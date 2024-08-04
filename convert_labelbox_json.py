import json
import yaml
import argparse
from os import listdir, makedirs
from os.path import isfile, join, exists


DATASET_YAML = '/data.yaml'
DATASET_JSON = '/data.json'
DATASET_DIR = 'datasets/'
TRAIN_IMAGES = '/train/images/'
VAL_IMAGES = '/val/images/'
TEST_IMAGES = '/test/images/'
TRAIN_LABELS = '/train/labels/'
VAL_LABELS = '/val/labels/'
TEST_LABELS = '/test/labels/'
PROJECT_ID = 'clyn59tv9063e07y07nku1bwx'

parser = argparse.ArgumentParser(description="Entraînement AI SONIA Vision")
# Choix du dataset
parser.add_argument('--dataset', 
                    type=str, 
                    required=True,
                    help='Nom du dataset')
parser.add_argument('--project-id', 
                    type=str, 
                    default=PROJECT_ID,
                    help='ID du projet')
args = parser.parse_args(namespace=None)

metadata = yaml.full_load(open(DATASET_DIR + args.dataset + DATASET_YAML, encoding='utf-8'))
class_names = list(metadata['names'].values())
class_ids = list(metadata['names'].keys())

train_img_path = DATASET_DIR+args.dataset+TRAIN_IMAGES
train_images = [f for f in listdir(train_img_path) if isfile(join(train_img_path, f))]

val_img_path = DATASET_DIR+args.dataset+VAL_IMAGES
val_images = [f for f in listdir(val_img_path) if isfile(join(val_img_path, f))]

test_img_path = DATASET_DIR+args.dataset+TEST_IMAGES
test_images = [f for f in listdir(test_img_path) if isfile(join(test_img_path, f))]

with open(DATASET_DIR + args.dataset + DATASET_JSON, encoding='utf-8') as f:
    data = json.load(f)

if not exists(DATASET_DIR+args.dataset+TEST_LABELS):
    makedirs(DATASET_DIR+args.dataset+TEST_LABELS)
if not exists(DATASET_DIR+args.dataset+TRAIN_LABELS):
    makedirs(DATASET_DIR+args.dataset+TRAIN_LABELS)
if not exists(DATASET_DIR+args.dataset+VAL_LABELS):
    makedirs(DATASET_DIR+args.dataset+VAL_LABELS)

for image in data:
    img_name = image['data_row']['external_id']
    img_width = image['media_attributes']['width']
    img_height = image['media_attributes']['height']
    label_name = img_name[:-3]+'txt'

    if img_name in test_images:
        label_path = DATASET_DIR+args.dataset+TEST_LABELS+label_name
    elif img_name in val_images:
        label_path = DATASET_DIR+args.dataset+VAL_LABELS+label_name
    else:
        label_path = DATASET_DIR+args.dataset+TRAIN_LABELS+label_name

    with open(label_path, 'w', encoding='utf-8') as label_file:
        for label in image['projects'][args.project_id]['labels'][0]['annotations']['objects']:
            class_id = class_ids[class_names.index(label['name'])]
            top = int(label['bounding_box']['top'])
            left = int(label['bounding_box']['left'])
            h = label['bounding_box']['height']
            w = label['bounding_box']['width']
            c_x = int(left + w/2)
            c_y = int(top + h/2)
            label_file.write('{} {} {} {} {}\n'.format(class_id, 
                                                       c_x/img_width, 
                                                       c_y/img_height, 
                                                       w/img_width, 
                                                       h/img_height))
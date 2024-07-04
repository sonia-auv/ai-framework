import json
import yaml
import argparse
from os import listdir
from os.path import isfile, join


DATASET_YAML = '/data.yaml'
DATASET_DIR = 'datasets/'
TRAIN_IMAGES = '/train/images/'
VAL_IMAGES = '/val/images/'
TEST_IMAGES = '/test/images/'
TRAIN_LABELS = '/train/labels/'
VAL_LABELS = '/val/labels/'
TEST_LABELS = '/test/labels/'


parser = argparse.ArgumentParser(description="Entraînement AI SONIA Vision")
# Choix du dataset
parser.add_argument('--dataset', 
                    type=str, 
                    required=True,
                    help='Nom du dataset')
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

with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

for image in data:
    img_name = image['data_row']['external_id']
    label_name = img_name[:-3]+'txt'

    if img_name in test_images:
        label_path = DATASET_DIR+args.dataset+TEST_LABELS+label_name
    elif img_name in val_images:
        label_path = DATASET_DIR+args.dataset+VAL_LABELS+label_name
    else:
        label_path = DATASET_DIR+args.dataset+TRAIN_LABELS+label_name

    with open(label_path, 'w', encoding='utf-8') as label_file:
        for label in image['projects']['cly4ximqr01kv081v0jqweg8a']['labels'][0]['annotations']['objects']:
            class_id = class_ids[class_names.index(label['name'])]
            x_min = int(label['bounding_box']['top'])
            y_min = int(label['bounding_box']['left'])
            h = label['bounding_box']['height']
            w = label['bounding_box']['width']
            x_max = int(x_min + h)
            y_max = int(y_min + w)
            label_file.write('{} {} {} {} {}\n'.format(class_id, x_min, y_min, x_max, y_max))
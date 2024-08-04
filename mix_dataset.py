import argparse
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copy
from utils import make_data_yaml

DATASET_DIR = 'datasets/'
DATASET_SUB_DIRS = ['/train/images/',
                    '/val/images/',
                    '/test/images/',
                    '/train/labels/',
                    '/val/labels/',
                    '/test/labels/']

parser = argparse.ArgumentParser(description="Mélange de datasets AI SONIA Vision")
# Choix du dataset
parser.add_argument('--new-dataset', 
                    type=str, 
                    required=True,
                    help='Nom du dataset')
parser.add_argument('--datasets',
                    '--list', 
                    nargs='+', 
                    required=True,
                    help='Datasets à mélanger')
args = parser.parse_args(namespace=None)

assert not exists(DATASET_DIR+args.new_dataset)
for dataset in args.datasets:
    assert exists(DATASET_DIR+dataset)

makedirs(DATASET_DIR+args.new_dataset)
for sub_dir in DATASET_SUB_DIRS:
    makedirs(DATASET_DIR+args.new_dataset+sub_dir)

for dataset in args.datasets:
    for sub_dir in DATASET_SUB_DIRS:
        path = DATASET_DIR+dataset+sub_dir
        for file in [f for f in listdir(path) if isfile(join(path, f))]:
            copy(path+file, DATASET_DIR+args.new_dataset+sub_dir+dataset+'_'+file)

from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copy

DATASET_YAML = '/data.yaml'
DATASET_DIR = 'datasets/'
DATASET_SUB_DIRS = ['/train/images/',
                    '/val/images/',
                    '/test/images/',
                    '/train/labels/',
                    '/val/labels/',
                    '/test/labels/']


def make_data_yaml(dataset_name):
    text = 'path: ../datasets/{}\ntrain: train/images/\nval: val/images/\ntest: test/images/\n\nnames:\n0: bin\n1: bin_bleue\n2: bin_rouge\n3: gate\n4: gate_rouge\n5: gate_bleue\n6: torpille\n7: torpille_target\n8: buoy\n9: samples_table\n10: path'.format(dataset_name)
    with open(DATASET_DIR+dataset_name+DATASET_YAML, "a") as f:
        f.write(text)


def get_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_dirs(path):
    return [f for f in listdir(path) if not isfile(join(path, f))]


def create_filename(name, number, lenght=4):
    nb_zeros = lenght - len(str(number))
    temp = ''
    for _ in range(nb_zeros):
        temp += '0'
    return name+'_'+temp+str(number)+'.jpg'


def find_name(path, name):
    number = 0
    filename = create_filename(name, number)
    while not filename_available(path+filename):
        number += 1
        filename = create_filename(name, number)
    return filename


def filename_available(path):
    return not exists(path)

def mix_datasets(new_dataset, dataset_list):
    assert not exists(DATASET_DIR+new_dataset)
    for dataset in dataset_list:
        assert exists(DATASET_DIR+dataset)

    makedirs(DATASET_DIR+new_dataset)
    for sub_dir in DATASET_SUB_DIRS:
        makedirs(DATASET_DIR+new_dataset+sub_dir)

    for dataset in dataset_list:
        for sub_dir in DATASET_SUB_DIRS:
            path = DATASET_DIR+dataset+sub_dir
            for file in [f for f in listdir(path) if isfile(join(path, f))]:
                copy(path+file, DATASET_DIR+new_dataset+sub_dir+dataset+'_'+file)
from os import mkdir
from os.path import exists
from shutil import move
from random import randint
from utils import make_data_yaml, get_dirs, get_files, find_name

TRAIN_PATH = 'train/images/'
VAL_PATH = 'val/images/'
TEST_PATH = 'test/images/'

class DatasetSonia():
    def __init__(self, src, dataset_name, train_proba, val_proba):
        assert train_proba+ val_proba <= 1
        self.src = src
        self.out = 'datasets/'+dataset_name+'/'
        self.train_proba = train_proba
        self.val_proba = val_proba
        self.dataset_name = dataset_name

    def split_dataset(self):
        if not exists(self.out+TRAIN_PATH):
            mkdir(self.out+'train')
            mkdir(self.out+'train/images')
        if not exists(self.out+VAL_PATH):
            mkdir(self.out+'val')
            mkdir(self.out+'val/images')
        if not exists(self.out+TEST_PATH):
            mkdir(self.out+'test')
            mkdir(self.out+'test/images')
        for dir in get_dirs(self.src):
            name = dir
            temp_files = []
            for file in get_files(self.out):
                if name in file:
                    temp_files.append(file)
            train = int(self.train_proba * len(temp_files))
            val = int(self.val_proba * len(temp_files))
            for _ in range(train):
                k = randint(0, len(temp_files)-1)
                file = temp_files.pop(k)
                move(self.out+file, self.out+TRAIN_PATH+file)
            for _ in range(val):
                k = randint(0, len(temp_files)-1)
                file = temp_files.pop(k)
                move(self.out+file, self.out+VAL_PATH+file)
            for file in temp_files:
                move(self.out+file, self.out+TEST_PATH+file)

    def move_files(self):
        for dir in get_dirs(self.src):
            files = get_files(self.src+dir)
            while len(files) > 0:
                k = randint(0, len(files)-1)
                out_name = find_name(self.out, dir)
                move(self.src+dir+'/'+files[k], self.out+out_name)
                files = get_files(self.src+dir)
    
    def create(self):
        mkdir(self.out)
        self.move_files()
        self.split_dataset()
        make_data_yaml(self.dataset_name)
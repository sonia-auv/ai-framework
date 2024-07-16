from os import listdir, mkdir
from os.path import isfile, join, exists
from shutil import move
from random import randint
import argparse


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


def move_files(src_path, out_path):
    for dir in get_dirs(src_path):
        files = get_files(src_path+dir)
        while len(files) > 0:
            k = randint(0, len(files)-1)
            out_name = find_name(out_path, dir)
            move(src_path+dir+'/'+files[k], out_path+out_name)
            files = get_files(src_path+dir)


def split_dataset(src_path, out_path, train_proba=0.7, val_proba=0.2):
    train_path = 'train/images/'
    val_path = 'val/images/'
    test_path = 'test/images/'
    if not exists(out_path+train_path):
        mkdir(out_path+'train')
        mkdir(out_path+'train/images')
    if not exists(out_path+val_path):
        mkdir(out_path+'val')
        mkdir(out_path+'val/images')
    if not exists(out_path+test_path):
        mkdir(out_path+'test')
        mkdir(out_path+'test/images')
    for dir in get_dirs(src_path):
        name = dir
        temp_files = []
        for file in get_files(out_path):
            if name in file:
                temp_files.append(file)
        train = int(train_proba * len(temp_files))
        val = int(val_proba * len(temp_files))
        for _ in range(train):
            k = randint(0, len(temp_files)-1)
            file = temp_files.pop(k)
            move(out_path+file, out_path+train_path+file)
        for _ in range(val):
            k = randint(0, len(temp_files)-1)
            file = temp_files.pop(k)
            move(out_path+file, out_path+val_path+file)
        for file in temp_files:
            move(out_path+file, out_path+test_path+file)



def main():
    parser = argparse.ArgumentParser(description="Dataset creation AI SONIA Vision")
    # Choix du modèle
    parser.add_argument('--src', 
                        type=str, 
                        required=True, 
                        help='Source dataset path (required)')
    parser.add_argument('--dataset-name', 
                        type=str, 
                        required=False,
                        default='on_working_directory', 
                        help='Dataset name (default:on_working_directory)')
    parser.add_argument('--train-proba', 
                        type=float, 
                        required=False, 
                        default=0.7, 
                        help='Proportion of train (default:0.7)')
    parser.add_argument('--val-proba', 
                        type=float, 
                        required=False, 
                        default=0.2, 
                        help='Proportion of val (default:0.2)')
    args = parser.parse_args(namespace=None)
    src_path = args.src
    out_path = 'datasets/'+args.dataset_name+'/'
    mkdir(out_path)
    move_files(src_path, out_path)
    split_dataset(src_path, out_path, args.train_proba, args.val_proba)
    


if __name__ == '__main__':
    main()
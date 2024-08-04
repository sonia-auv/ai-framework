import json
import yaml
import labelbox
from os import listdir, makedirs
from os.path import isfile, join, exists

API_KEY = ''
PROJECT_ID = ''

API_KEY_PATH = './config/.labelbox_api_key.txt'
DATASET_YAML = '/data.yaml'
DATASET_JSON = '/data.json'
DATASET_DIR = 'datasets/'
TRAIN_IMAGES = '/train/images/'
VAL_IMAGES = '/val/images/'
TEST_IMAGES = '/test/images/'
TRAIN_LABELS = '/train/labels/'
VAL_LABELS = '/val/labels/'
TEST_LABELS = '/test/labels/'


class LabelingSonia():
    def __init__(self, dataset, project_id):
        self.project_id = project_id
        self.dataset = dataset
        self.__set_api_key()
        self.client = labelbox.Client(api_key=API_KEY)
        self.project = self.client.get_project(self.project_id)

    def __set_api_key(self):
        if exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r', encoding='UTF-8') as f:
                self.api_key = f.read()
        else:
            self.api_key = API_KEY

    def load_labels(self):
        export_task = self.project.export_v2(params={})
        export_task.wait_till_done()
        if export_task.errors:
            print(export_task.errors)
        else:
            with open(DATASET_DIR+self.dataset+'/data.json', 'w', encoding='utf-8') as f:
                json.dump(export_task.result, f, ensure_ascii=False, indent=4)

    def convert_labels(self):
        self.load_yaml()
        self.load_labels()
        self.load_json()
        self.load_images()
        self.make_labels_dirs()
        self.convert_json()

    def load_yaml(self):
        with open(DATASET_DIR + self.dataset + DATASET_YAML, encoding='utf-8') as f:
            metadata = yaml.full_load(f)
            self.class_names = list(metadata['names'].values())
            self.class_ids = list(metadata['names'].keys())

    def load_json(self):
        with open(DATASET_DIR + self.dataset + DATASET_JSON, encoding='utf-8') as f:
            self.data = json.load(f)

    def load_images(self):
        train_img_path = DATASET_DIR+self.dataset+TRAIN_IMAGES
        val_img_path = DATASET_DIR+self.dataset+VAL_IMAGES
        test_img_path = DATASET_DIR+self.dataset+TEST_IMAGES
        
        self.train_images = [f for f in listdir(train_img_path) if isfile(join(train_img_path, f))]
        self.val_images = [f for f in listdir(val_img_path) if isfile(join(val_img_path, f))]
        self.test_images = [f for f in listdir(test_img_path) if isfile(join(test_img_path, f))]

    def make_labels_dirs(self):
        if not exists(DATASET_DIR+self.dataset+TEST_LABELS):
            makedirs(DATASET_DIR+self.dataset+TEST_LABELS)
        if not exists(DATASET_DIR+self.dataset+TRAIN_LABELS):
            makedirs(DATASET_DIR+self.dataset+TRAIN_LABELS)
        if not exists(DATASET_DIR+self.dataset+VAL_LABELS):
            makedirs(DATASET_DIR+self.dataset+VAL_LABELS)

    def get_label_path(self, image):
        img_name = image['data_row']['external_id']
        label_name = img_name[:-3]+'txt'
        if img_name in self.test_images:
            return DATASET_DIR+self.dataset+TEST_LABELS+label_name
        elif img_name in self.val_images:
            return DATASET_DIR+self.dataset+VAL_LABELS+label_name
        else:
            return DATASET_DIR+self.dataset+TRAIN_LABELS+label_name

    def create_label_file(self, image, label_path):
        img_width = image['media_attributes']['width']
        img_height = image['media_attributes']['height']
        with open(label_path, 'w', encoding='utf-8') as label_file:
            for label in image['projects'][self.project_id]['labels'][0]['annotations']['objects']:
                class_id = self.class_ids[self.class_names.index(label['name'])]
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

    def convert_json(self):
        for image in self.data:
            self.create_label_file(image, self.get_label_path(image))

from os import makedirs
from os.path import exists
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import config.credentials as credentials
import labelbox as lb
import cv2


def get_dataset(dataset_name):
    datasets = client.get_datasets()
    print(f"Searching for dataset: {dataset_name}")
    for dataset in datasets:
        if dataset.name == dataset_name:
            print(f"Dataset '{dataset_name}' found.")
            return dataset
        else:
            print(f"Dataset '{dataset_name}' not found.")
            return None

dataset_name = "rosobub-2025"
client = lb.Client(api_key=credentials.API_KEY)
            
dataset = get_dataset(dataset_name)
data_rows = dataset.data_rows()
if not exists("./temp_update/"):
    makedirs("./temp_update")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for data_row in data_rows:
    print(f"Global key: {data_row.external_id}")
    response = requests.get(data_row.row_data)
    img = Image.open(BytesIO(response.content))
    ycrcb_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y_clahe = clahe.apply(y)
    clahe_ycrcb = cv2.merge([y_clahe, cr, cb])
    clahe_img = cv2.cvtColor(clahe_ycrcb, cv2.COLOR_YCrCb2BGR)
    img_path = f"./temp_update/{data_row.external_id}.jpg"
    cv2.imwrite(img_path, clahe_img)
    data_row.update(
        row_data=img_path
    )
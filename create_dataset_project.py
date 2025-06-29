import labelbox as lb
import config.credentials as credentials
import labelbox.types as lb_types
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from ultralytics import YOLO
from uuid import uuid4


API_KEY = credentials.API_KEY


def get_project(project_name, client):
    for project in client.get_projects():
        if project.name == project_name:
            return project.uid
    return None


def available_classes(project):
    available_classes = []
    for tool in project.ontology().normalized['tools']:
        available_classes.append(tool['name'])


def make_project(client, project_name, ontology_name):
    project_id = get_project(project_name, client) 
    if project_id is not None:
        print(f"Project '{project_name}' already exists.")
        return client.get_project(project_id)
    
    
    ontology = client.get_ontologies(name_contains=ontology_name).get_one()

    project = client.create_project(
        name = PROJECT_NAME,
        description = '',
        media_type = lb.MediaType.Image
    )
    project.connect_ontology(ontology=ontology)

    return project


def make_dataset(client, dataset_name):
    datasets = client.get_datasets()
    for dataset in datasets:
        if dataset.name == dataset_name:
            print(f"Dataset '{dataset_name}' already exists.")
            return dataset
    
    return client.create_dataset(
        name=dataset_name,
        iam_integration=None
        )



def batch_exists(batches, batch_name):
    for batch in batches:
        if batch.name == batch_name:
            return True
    return False


def make_data_rows(images_files):
    data_rows = []
    for image_file in images_files:
        data_rows.append(
            {'row_data': image_file,
             'global_key': image_file}
        )
    return data_rows


def create_data_rows(dataset, data_rows):
    existing_data_rows = dataset.data_rows()
    existing_uuids = []
    for existing_row in existing_data_rows:
        existing_uuids.append(existing_row.external_id)
    new_data_rows = [row for row in data_rows if row['row_data'] not in existing_uuids]
    
    if new_data_rows:
        task_will_succeed = dataset.create_data_rows(new_data_rows)
        task_will_succeed.wait_till_done()
        print(f"Created {len(new_data_rows)} new data rows.")
    else:
        print("No new data rows to create.")
    return [row['global_key'] for row in new_data_rows]


def make_batch(project, global_keys):
    batches = project.batches()
    count = 1
    batch_name = f"batch_{count}"
    while batch_exists(batches, batch_name):
        count += 1
        batch_name = f"batch_{count}"
    return project.create_batch(name=batch_name,
                                global_keys=global_keys,
                                priority=1)


def predict_image_labels(img_path, model):
    image = cv2.imread(img_path)
    results = model.predict(image, conf=0.5, verbose=False)
    annotations = []
    for prediction in results:
        for i in range(prediction.boxes.shape[0]):
            cls = int(prediction.boxes.cls[i].item())
            name = prediction.names[cls]
            x1, y1, x2, y2 = prediction.boxes.xyxy[i].cpu().numpy()
            annotations.append(lb_types.ObjectAnnotation(
                name=name,
                value=lb_types.Rectangle(
                    start=lb_types.Point(x=int(round(x1)), y=int(round(y1))),
                    end=lb_types.Point(x=int(round(x2)), y=int(round(y2))),
                    )
                )
            )
    return annotations


def upload_labels(client, project, model, global_keys):
    labels = []
    for key in global_keys:
        annotations = predict_image_labels(key, model)
        labels.append(
            lb_types.Label(data={"global_key": key},
                        annotations = annotations
            )
        )

    # Upload MAL label for this data row in project
    upload_job = lb.MALPredictionImport.create_from_objects(
        client = client, 
        project_id = project.uid, 
        name="mal_job"+str(uuid4()), 
        predictions=labels
    )

    if len(upload_job.errors) == 0:
        print("Upload job completed successfully.")
    else:
        print("Upload job completed with errors.")
        print(f"Errors: {upload_job.errors}")


PROJECT_NAME = "photos-local"
ONTOLOGY_NAME = 'Robosub-2025-box'
DATASET_NAME = "photos-local-dataset"
INPUT_IMAGES_DIR = "source/photos-local"
MODEL_PATH = None 


client = lb.Client(api_key=API_KEY)


project = make_project(client, PROJECT_NAME, ONTOLOGY_NAME)


dataset = make_dataset(client, DATASET_NAME)

# batch = make_batch(project)
images_files = [join(INPUT_IMAGES_DIR, f) for f in listdir(INPUT_IMAGES_DIR) if isfile(join(INPUT_IMAGES_DIR, f))]

data_rows = make_data_rows(images_files)

global_keys = create_data_rows(dataset, data_rows)

print("Data rows uploaded")

make_batch(project, global_keys)

if MODEL_PATH is not None:
    upload_labels(client, project, YOLO(MODEL_PATH), global_keys)
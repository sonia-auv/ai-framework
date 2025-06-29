import labelbox
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import json
import os
from PIL import Image
import requests
from io import BytesIO
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from PIL import ImageDraw
from PIL import ImageFont
import config.credentials as credentials

API_KEY = credentials.API_KEY

def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data.tolist(), f)


def create_yaml_content(project_name, count, available_classes):
    yaml_content = f'path: ../{os.path.join("datasets", project_name + "_" + str(count))}\n'
    yaml_content += 'train: train/images\n'
    yaml_content += 'val: val/images\n'
    yaml_content += 'test: test/images\n\n'
    yaml_content += f'nc: {len(available_classes)}\n'
    yaml_content += 'names:\n'
    for i, class_name in enumerate(available_classes):
        yaml_content += f'  {i}: {class_name}\n'
    return yaml_content


def make_dataset_dir(project_name, available_classes):
    count = 1
    path = os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count))
    while os.path.exists(path):
        count += 1
        path = os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count))
    os.makedirs(path)
    os.makedirs(os.path.join(path, 'train', 'images'))
    os.makedirs(os.path.join(path, 'train', 'labels'))
    os.makedirs(os.path.join(path, 'train', 'image_with_labels'))
    os.makedirs(os.path.join(path, 'val', 'images'))
    os.makedirs(os.path.join(path, 'val', 'labels'))
    os.makedirs(os.path.join(path, 'val', 'image_with_labels'))
    os.makedirs(os.path.join(path, 'test', 'images'))
    os.makedirs(os.path.join(path, 'test', 'labels'))
    os.makedirs(os.path.join(path, 'test', 'image_with_labels'))
    with open(os.path.join(path, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(create_yaml_content(project_name, count, available_classes))

    return path


def get_prev_dataset(project_name):
    count = 1
    path = os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count))
    while os.path.exists(path):
        count += 1
        path = os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count))
    
    if os.path.exists(os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count - 1))):
        return os.path.join(os.getcwd(), 'datasets', project_name + '_' + str(count - 1))
    else:
        return None


def box_to_obb(box):
    left = box['left']
    top = box['top']
    width = box['width']
    height = box['height']
    center_x = left + width / 2
    center_y = top + height / 2
    return {
        'cx': center_x,
        'cy': center_y,
        'w': width,
        'h': height,
        'angle': 0.0  # Assuming no rotation for simplicity
    }


def polygon_to_box(polygon):
    x_coords = [point['x'] for point in polygon]
    y_coords = [point['y'] for point in polygon]
    left = min(x_coords)
    top = min(y_coords)
    right = max(x_coords)
    bottom = max(y_coords)
    width = right - left
    height = bottom - top
    return {
        'left': left,
        'top': top,
        'width': width,
        'height': height
    }


def polygon_to_obb(polygon):
    polygon_points = [(point['x'], point['y']) for point in polygon]
    poly = Polygon(polygon_points)
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]  # last point is same as first

    # Calculate center, width, height, and angle
    cx = sum([p[0] for p in rect_coords]) / 4
    cy = sum([p[1] for p in rect_coords]) / 4
    edge1 = np.array(rect_coords[1]) - np.array(rect_coords[0])
    edge2 = np.array(rect_coords[2]) - np.array(rect_coords[1])
    width = np.linalg.norm(edge1)
    height = np.linalg.norm(edge2)
    angle = np.degrees(np.arctan2(edge1[1], edge1[0]))

    return {
        'cx': cx,
        'cy': cy,
        'w': width,
        'h': height,
        'angle': angle
    }


def mask_to_box(mask):
    # Convert mask to bounding box
    mask_array = np.array(mask)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    width = right - left + 1
    height = bottom - top + 1
    return {
        'left': int(left),
        'top': int(top),
        'width': int(width),
        'height': int(height)
    }


def mask_to_obb(mask):
    mask_array = np.array(mask)
    coords = np.column_stack(np.where(mask_array))
    if coords.shape[0] == 0:
        return None
    points = [tuple(coord[::-1]) for coord in coords]  # (x, y)
    poly = MultiPoint(points).convex_hull
    min_rect = poly.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]
    cx = sum([p[0] for p in rect_coords]) / 4
    cy = sum([p[1] for p in rect_coords]) / 4
    edge1 = np.array(rect_coords[1]) - np.array(rect_coords[0])
    edge2 = np.array(rect_coords[2]) - np.array(rect_coords[1])
    width = np.linalg.norm(edge1)
    height = np.linalg.norm(edge2)
    angle = np.degrees(np.arctan2(edge1[1], edge1[0]))
    return {
        'cx': cx,
        'cy': cy,
        'w': width,
        'h': height,
        'angle': angle
    }



def manage_detected_item(item, label_type, client):
    if 'bounding_box' in item.keys():
        if label_type == 'box':
            return item['bounding_box']
        elif label_type == 'obb':
            return box_to_obb(item['bounding_box'])
        elif label_type == 'mask':
            raise ValueError("Mask output is not supported for box or obb labels.")
    elif 'polygon' in item.keys():
        if label_type == 'box':
            return polygon_to_box(item['polygon'])
        elif label_type == 'obb':
            return polygon_to_obb(item['polygon'])
        elif label_type == 'mask':
            raise ValueError("Mask output is not supported for box or obb labels.")
    elif 'mask' in item.keys():
        response = requests.get(item['mask']['url'], headers=client.headers)
        mask = Image.open(BytesIO(response.content))
        if label_type == 'box':
            return mask_to_box(mask)
        elif label_type == 'obb':
            return mask_to_obb(mask)
        elif label_type == 'mask':
            return item['mask']['url']


def download_json(project):
    export_task = project.export()
    export_task.wait_till_done()
    export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start()
    stream = export_task.get_buffered_stream()
    return [data_row.json for data_row in stream]


def get_project(project_name, client):
    for project in client.get_projects():
        if project.name == project_name:
            return project.uid
    return None


# label_type = 'box'  or 'obb' or 'mask' 
def import_dataset(api_key, project_name, label_type='box', download = True):
    client = labelbox.Client(api_key)
    project_id = get_project(project_name, client)

    project = client.get_project(project_id)
    if project is None:
        exit("Project not found. Please check the project name.")

    available_classes = []
    for tool in project.ontology().normalized['tools']:
        available_classes.append(tool['name'])

    prev_dataset_path = get_prev_dataset(project_name)
    dataset_path = make_dataset_dir(project_name, available_classes)

    print('Importing json')
    if download or prev_dataset_path is None or not os.path.exists(os.path.join(prev_dataset_path, "export_json.ndjson")):
        export_json = download_json(project)
    else:
        with open(os.path.join(prev_dataset_path, "export_json.ndjson"), "r", encoding="utf-8") as f:
            export_json = [json.loads(line) for line in f]

    # Save export_json as NDJSON file
    with open(os.path.join(dataset_path, "export_json.ndjson"), "w", encoding="utf-8") as f:
        for entry in export_json:
            f.write(json.dumps(entry) + "\n")
    
    print('Json imported')
    raw_dataset = []
    present_classes = []
    for data_row in export_json:
        img_name = data_row['data_row']['id']
        img_url = data_row['data_row']['row_data']
        names = []
        detected_element = []
        if len(data_row['projects'][project_id]['labels']) > 0:
            for item in data_row['projects'][project_id]['labels'][0]['annotations']['objects']:
                names.append(item['name'])
                detected_element.append(manage_detected_item(item, label_type, client)) # TESTER LES DIFFÉRENTS TYPES DE LABELS
            raw_dataset.append([img_name, img_url, {'names': names, 'detection': detected_element}])
        else:
            raw_dataset.append([img_name, img_url, {'names': None, 'detection': None}])
        classes = np.zeros(len(available_classes), dtype=int)
        for name in names:
            if name in available_classes:
                classes[available_classes.index(name)] = 1
        present_classes.append(classes)

    raw_dataset = np.array(raw_dataset)
    present_classes = np.array(present_classes)
    train_set, _, val_set, val_classes = iterative_train_test_split(raw_dataset, present_classes, .25)
    val_set, _, test_set, _ = iterative_train_test_split(val_set, val_classes, .1)

    save_json('./train_set.json', train_set)
    save_json('./val_set.json', val_set)

    return train_set, val_set, test_set, available_classes, dataset_path, client


def save_img_with_boxes(img, boxes, names, image_with_labels_path):
    for i in range(len(boxes)):
        top = boxes[i]['top']
        left = boxes[i]['left']
        height = boxes[i]['height']
        width = boxes[i]['width']
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([left, top, left + width, top + height], outline="red", width=2)
        # Write class name on the rectangle
        font_size = 16
        text_y = top - font_size if top - font_size > 0 else top + 2
        draw.text((left, text_y), names[i], fill="red", font=ImageFont.load_default())
        
    with open(image_with_labels_path, 'w', encoding='utf-8') as f:
        Image.Image.save(img, f, format='JPEG')


def save_boxes(img, boxes, names, labels_path, available_classes):
    label_file_content = ''
    img_width, img_height = img.size
    for i in range(len(boxes)):
        top = boxes[i]['top']
        left = boxes[i]['left']
        height = boxes[i]['height']
        width = boxes[i]['width']

        label_file_content += f'{available_classes.index(names[i])} {(left + (width / 2)) / img_width} {(top + (height / 2)) / img_height} {width / img_width} {height / img_height}\n'
    
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write(label_file_content)


def save_img_with_obb(img, obbs, names, image_with_labels_path):
    for i in range(len(obbs)):
        cx = obbs[i]['cx']
        cy = obbs[i]['cy']
        w = obbs[i]['w']
        h = obbs[i]['h']
        angle = obbs[i]['angle']

        # Draw OBB on image
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = w / 2
        dy = h / 2
        # Four corners relative to center
        corners = [
            (-dx, -dy),
            (dx, -dy),
            (dx, dy),
            (-dx, dy)
        ]
        # Rotate and translate corners
        box_points = []
        for x, y in corners:
            x_rot = cos_t * x - sin_t * y + cx
            y_rot = sin_t * x + cos_t * y + cy
            box_points.append((x_rot, y_rot))
        draw = ImageDraw.Draw(img)
        draw.polygon(box_points, outline="blue", width=2)
        # Write class name near the first corner of the OBB
        font_size = 16
        text_x, text_y = box_points[0]
        draw.text((text_x, text_y - font_size if text_y - font_size > 0 else text_y + 2), names[i], fill="blue", font=ImageFont.load_default())

    with open(image_with_labels_path, 'w', encoding='utf-8') as f:
        Image.Image.save(img, f, format='JPEG')


def save_obb(img, obbs, names, labels_path, available_classes):
    label_file_content = ''
    img_width, img_height = img.size
    for i in range(len(obbs)):
        cx = obbs[i]['cx']
        cy = obbs[i]['cy']
        w = obbs[i]['w']
        h = obbs[i]['h']
        angle = obbs[i]['angle']
        class_idx = available_classes.index(names[i])

        # Normalize values for YOLO OBB format: class cx cy w h angle
        label_file_content += f"{class_idx} {cx / img_width} {cy / img_height} {w / img_width} {h / img_height} {angle}\n"

    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write(label_file_content)


def save_img_with_mask(img, masks, names, image_with_labels_path, client):
    color_map = {
        name: (
            (hash(name) & 0xFF), 
            ((hash(name) >> 8) & 0xFF), 
            ((hash(name) >> 16) & 0xFF), 
            200  # alpha
        )
        for name in set(names)
    }

    for i in range(len(masks)):
        response = requests.get(masks[i], headers=client.headers)
        mask = Image.open(BytesIO(response.content))

        # Resize mask if needed
        if mask.size != img.size:
            mask = mask.resize(img.size, resample=Image.NEAREST)

        # Save mask as binary (0 or 1)
        mask_array = np.array(mask) > 0
        # Find bounding box for YOLO format
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        if not rows.any() or not cols.any():
            continue
        top, _ = np.where(rows)[0][[0, -1]]
        left, _ = np.where(cols)[0][[0, -1]]
        mask_rgba = Image.fromarray(np.uint8(mask_array) * 255, mode="L").convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        color = color_map.get(names[i], (0, 255, 0, 200))  # fallback to green
        for y in range(mask_rgba.height):
            for x in range(mask_rgba.width):
                if mask_array[y, x]:
                    overlay.putpixel((x, y), color)
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay).convert("RGB")
        # Write class name at the top-left of the mask
        draw = ImageDraw.Draw(img)
        font_size = 16
        draw.text((left, top - font_size if top - font_size > 0 else top + 2), names[i], fill=color[:3], font=ImageFont.load_default())

    with open(image_with_labels_path, 'w', encoding='utf-8') as f:
        Image.Image.save(img, f, format='JPEG')


def save_mask(img, masks, names, labels_path):
    # img_width, img_height = img.size
    # for i in range(len(masks)):
    #     masks

    # with open(labels_path, 'w', encoding='utf-8') as f:
    #     f.write(label_file_content)
    # TODO
    pass


def save_set_to_dir(dataset, set_type, available_classes, dataset_path, label_type, client):
    for k, item in enumerate(dataset):
        print('{:02f}%'.format(100 * k / len(dataset)))
        img_name = item[0]
        img_url = item[1]
        labels = item[2]
        img_path = os.path.join(dataset_path, set_type, 'images', img_name + '.jpg')
        image_with_labels_path = os.path.join(dataset_path, set_type, 'image_with_labels', img_name + '.jpg')
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))

        with open(img_path, 'w', encoding='utf-8') as f:
            Image.Image.save(img, f, format='JPEG')

        if labels["detection"] is not None:
            labels_path = os.path.join(dataset_path, set_type, 'labels', img_name + '.txt')
            if label_type == 'box':
                save_img_with_boxes(img, labels["detection"], labels["names"], image_with_labels_path)
                save_boxes(img, labels["detection"], labels["names"], labels_path, available_classes)

            elif label_type == 'obb':
                save_img_with_obb(img, labels["detection"], labels["names"], image_with_labels_path)
                save_obb(img, labels["detection"], labels["names"], labels_path, available_classes)

            elif label_type == 'mask':
                save_img_with_mask(img, labels["detection"], labels["names"], image_with_labels_path, client)
                save_mask(img, labels["detection"], labels["names"], labels_path)



dataset_name = 'Robosub-2025'
label_type = 'mask'
download = False

print("Importing dataset...")
train_set, val_set, test_set, available_classes, dataset_path, client = import_dataset(API_KEY, dataset_name, label_type, download)
print("Dataset imported successfully.")

print("Saving train set to directory...")
save_set_to_dir(train_set, 'train', available_classes, dataset_path, label_type, client)

print("Saving validation sets to directory...")
save_set_to_dir(val_set, 'val', available_classes, dataset_path, label_type, client)

print("Saving test set to directory...")
save_set_to_dir(test_set, 'test', available_classes, dataset_path, label_type, client)
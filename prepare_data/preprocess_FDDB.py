import numpy as np
import json
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
import random


def ellipse_to_box(major_axis_radius, minor_axis_radius, angle, center_x, center_y):
    half_h = major_axis_radius * np.sin(-angle)
    half_w = minor_axis_radius * np.sin(-angle)
    xmin, xmax = center_x - half_w, center_x + half_w
    ymin, ymax = center_y - half_h, center_y + half_h
    return xmin, ymin, xmax, ymax


def get_boxes(path):
    with open(path, 'r') as f:
        content = f.readlines()
        content = [s.strip() for s in content]

    boxes = {}
    num_lines = len(content)
    i = 0
    name = None

    while i < num_lines:
        s = content[i]
        if 'big/img' in s:
            if name is not None:
                assert len(boxes[name]) == num_boxes
            name = s + '.jpg'
            boxes[name] = []
            i += 1
            num_boxes = int(content[i])
            i += 1
        else:
            numbers = [float(f) for f in s.split(' ')[:5]]
            major_axis_radius, minor_axis_radius, angle, center_x, center_y = numbers

            xmin, ymin, xmax, ymax = ellipse_to_box(
                major_axis_radius, minor_axis_radius,
                angle, center_x, center_y
            )
            if xmin == xmax or ymin == ymax:
                num_boxes -= 1
            else:
                boxes[name].append((
                    min(xmin, xmax), min(ymin, ymax),
                    max(xmin, xmax), max(ymin, ymax)
                ))
            i += 1
    return boxes


def draw_boxes_on_image(path, boxes):

    image = Image.open(path)
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size

    for b in boxes:
        xmin, ymin, xmax, ymax = b

        fill = (255, 255, 255, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
    return image


def get_annotation(boxes, path, name, width, height):
    annotation = {
      "filename": name,
      "size": {"depth": 3, "width": width, "height": height}
    }
    objects = []
    for b in boxes[path]:
        xmin, ymin, xmax, ymax = b
        objects.append({"bndbox": {"ymin": ymin, "ymax": ymax, "xmax": xmax, "xmin": xmin}, "name": "face"})
    annotation["object"] = objects
    return annotation


def preprocess_FDDB(IMAGES_DIR, BOXES_DIR, RESULT_DIR):
    # collect paths to all images

    all_paths = []
    for path, subdirs, files in tqdm(os.walk(IMAGES_DIR)):
        for name in files:
            all_paths.append(os.path.join(path, name))

    metadata = pd.DataFrame(all_paths, columns=['full_path'])

    # strip root folder
    metadata['path'] = metadata.full_path.apply(lambda x: os.path.relpath(x, IMAGES_DIR))

    # all unique endings
    metadata.path.apply(lambda x: x.split('.')[-1]).unique()

    # number of images
    print('number of images', len(metadata))

    annotation_files = os.listdir(BOXES_DIR)
    annotation_files = [f for f in annotation_files if f.endswith('ellipseList.txt')]
    annotation_files = [os.path.join(BOXES_DIR, f) for f in annotation_files]

    boxes = {}
    for p in annotation_files:
        boxes.update(get_boxes(p))

    # check number of images with annotations
    # and number of boxes
    # (these values are taken from the official website)
    assert len(boxes) == 2845
    assert sum(len(b) for b in boxes.values()) == 5171 - 1  # one box is empty

    metadata = metadata.loc[metadata.path.apply(lambda x: x in boxes)]
    metadata = metadata.reset_index(drop=True)

    # =======================================================================
    i = random.randint(0, len(metadata) - 1)  # choose a random image
    some_boxes = boxes[metadata.path[i]]
    draw_boxes_on_image(metadata.full_path[i], some_boxes)

    # =======================================================================
    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    os.mkdir(RESULT_DIR)
    os.mkdir(os.path.join(RESULT_DIR, 'images'))
    os.mkdir(os.path.join(RESULT_DIR, 'annotations'))

    for T in tqdm(metadata.itertuples()):
        # get width and height of an image
        image = cv2.imread(T.full_path)
        h, w, c = image.shape
        assert c == 3

        # name of the image
        name = '-'.join(T.path.split('/')[:3]) + '_' + T.path.split('/')[-1]
        assert name.endswith('.jpg')

        # copy the image
        shutil.copy(T.full_path, os.path.join(RESULT_DIR, 'images', name))

        # save annotation for it
        d = get_annotation(boxes, T.path, name, w, h)
        json_name = name[:-4] + '.json'
        json.dump(d, open(os.path.join(RESULT_DIR, 'annotations', json_name), 'w'))


if __name__ == '__main__':
    IMAGES_DIR = '/home/lijc08/deeplearning/Data/FDDB/originalPics/'
    BOXES_DIR = '/home/lijc08/deeplearning/Data/FDDB/FDDB-folds/'
    RESULT_DIR = '/home/lijc08/deeplearning/Data/WIDER/val/'
    # IMAGES_DIR = '/home/gpu2/hdd/dan/FDDB/originalPics/'
    # BOXES_DIR = '/home/gpu2/hdd/dan/FDDB/FDDB-folds/'
    # RESULT_DIR = '/home/gpu2/hdd/dan/FDDB/val/'
    preprocess_FDDB(IMAGES_DIR, BOXES_DIR, RESULT_DIR)

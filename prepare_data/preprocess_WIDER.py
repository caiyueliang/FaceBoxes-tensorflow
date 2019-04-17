import numpy as np
import json
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
import random


def draw_boxes_on_image(path, boxes):

    image = Image.open(path)
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size

    for b in boxes:
        xmin, ymin, w, h = b
        xmax, ymax = xmin + w, ymin + h

        fill = (255, 255, 255, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
    return image


def get_annotation(boxes, path, width, height):
    name = path.split('/')[-1]
    annotation = {
      "filename": name,
      "size": {"depth": 3, "width": width, "height": height}
    }
    objects = []
    for b in boxes[path]:
        xmin, ymin, w, h = b
        xmax, ymax = xmin + w, ymin + h
        objects.append({
            "bndbox": {"ymin": ymin, "ymax": ymax, "xmax": xmax, "xmin": xmin},
            "name": "face"
        })
    annotation["object"] = objects
    return annotation


def preprocess_WIDER(IMAGES_DIR, BOXES_PATH, RESULT_DIR):
    # first run this script for this images:
    # IMAGES_DIR = '/home/lijc08/deeplearning/Data/WIDER/WIDER_train/images/'
    # BOXES_PATH = '/home/lijc08/deeplearning/Data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    # RESULT_DIR = '/home/lijc08/deeplearning/Data/WIDER/train/'
    # IMAGES_DIR = '/home/gpu2/hdd/dan/WIDER/WIDER_train/images/'
    # BOXES_PATH = '/home/gpu2/hdd/dan/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    # RESULT_DIR = '/home/gpu2/hdd/dan/WIDER/train/'

    # then run for this images:
    # IMAGES_DIR = '/home/gpu2/hdd/dan/WIDER/WIDER_val/images/'
    # BOXES_PATH = '/home/gpu2/hdd/dan/WIDER/wider_face_split/wider_face_val_bbx_gt.txt'
    # RESULT_DIR = '/home/gpu2/hdd/dan/WIDER/train_part2/'

    # collect paths to all images
    all_paths = []
    for path, subdirs, files in tqdm(os.walk(IMAGES_DIR)):
        for name in files:
            all_paths.append(os.path.join(path, name))

    metadata = pd.DataFrame(all_paths, columns=['full_path'])

    # strip root folder
    metadata['path'] = metadata.full_path.apply(lambda x: os.path.relpath(x, IMAGES_DIR))

    # see all unique endings
    metadata.path.apply(lambda x: x.split('.')[-1]).unique()

    # number of images
    print('number of images', len(metadata))

    # read annotations
    with open(BOXES_PATH, 'r') as f:
        content = f.readlines()
        content = [s.strip() for s in content]

    boxes = {}
    num_lines = len(content)
    print('num_lines', num_lines)
    i = 0
    name = None

    while i < num_lines:
        s = content[i]
        # print(s)
        if s.endswith('.jpg'):
            if name is not None:
                # print(boxes[name], num_boxes)
                assert len(boxes[name]) == num_boxes
            name = s
            boxes[name] = []
            i += 1
            num_boxes = int(content[i])
            i += 1
        else:
            xmin, ymin, w, h = s.split(' ')[:4]
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            if h <= 0 or w <= 0:
                print(name)
                # some boxes are weird!
                # so i don't use them
                num_boxes -= 1 if num_boxes > 0 else 0
            else:
                boxes[name].append((xmin, ymin, w, h))
            i += 1

    # check that all images have bounding boxes
    assert metadata.path.apply(lambda x: x in boxes).all()

    # =======================================================================
    i = random.randint(0, len(metadata) - 1)  # choose a random image
    some_boxes = boxes[metadata.path[i]]
    draw_boxes_on_image(metadata.full_path[i], some_boxes)

    # =======================================================================
    # create a folder for the converted dataset
    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    os.mkdir(RESULT_DIR)
    os.mkdir(os.path.join(RESULT_DIR, 'images'))
    os.mkdir(os.path.join(RESULT_DIR, 'annotations'))

    # =======================================================================
    for T in tqdm(metadata.itertuples()):
        # get width and height of an image
        image = cv2.imread(T.full_path)
        h, w, c = image.shape
        assert c == 3

        # name of the image
        name = T.path.split('/')[-1]
        assert name.endswith('.jpg')

        # copy the image
        shutil.copy(T.full_path, os.path.join(RESULT_DIR, 'images', name))

        # save annotation for it
        d = get_annotation(boxes, T.path, w, h)
        json_name = name[:-4] + '.json'
        json.dump(d, open(os.path.join(RESULT_DIR, 'annotations', json_name), 'w'))


if __name__ == '__main__':
    IMAGES_DIR = '/home/lijc08/deeplearning/Data/WIDER/WIDER_train/images/'
    BOXES_PATH = '/home/lijc08/deeplearning/Data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    RESULT_DIR = '/home/lijc08/deeplearning/Data/WIDER/train/'
    # # IMAGES_DIR = '/home/gpu2/hdd/dan/WIDER/WIDER_train/images/'
    # # BOXES_PATH = '/home/gpu2/hdd/dan/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    # # RESULT_DIR = '/home/gpu2/hdd/dan/WIDER/train/'
    preprocess_WIDER(IMAGES_DIR, BOXES_PATH, RESULT_DIR)

    # then run for this images:
    IMAGES_DIR = '/home/lijc08/deeplearning/Data/WIDER/WIDER_val/images/'
    BOXES_PATH = '/home/lijc08/deeplearning/Data/WIDER/wider_face_split/wider_face_val_bbx_gt.txt'
    RESULT_DIR = '/home/lijc08/deeplearning/Data/WIDER/train_part2/'
    # IMAGES_DIR = '/home/gpu2/hdd/dan/WIDER/WIDER_val/images/'
    # BOXES_PATH = '/home/gpu2/hdd/dan/WIDER/wider_face_split/wider_face_val_bbx_gt.txt'
    # RESULT_DIR = '/home/gpu2/hdd/dan/WIDER/train_part2/'
    preprocess_WIDER(IMAGES_DIR, BOXES_PATH, RESULT_DIR)


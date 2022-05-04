import os
from os.path import exists

import pickle
from random import random, sample

import numpy as np
from paz.abstract import ProcessingSequence, SequentialProcessor
from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentDetection
import paz.processors as pr
import paz.backend as P

from caltech_processor import AugmentCaltech


def deprocess_image(image):
    image = (image + pr.BGR_IMAGENET_MEAN).astype('uint8')
    return P.image.convert_color_space(image, pr.BGR2RGB)


def extract_box_caltech(file, width=640, height=480):
    """
    Extracts bounding boxes from the provided file.
    :param file: path to annotations file.
    :param width: image width. default is 640px (caltech pedestrian dataset size)
    :param height: image height. default is 480px (caltech pedestrian dataset size)
    :returns list of bounding boxes.
    """
    model_labels = {
        "background": 0,
        "person": 1,
        "person-fa": 1,
        "person?": 1,
        "people": 2
    }

    box_data = []
    with open(file, 'r') as f:
        line = f.readline()  # skip first line, its not a bounding box
        while line:
            line = f.readline()  # get line

            if not line:  # exit condition
                break

            split = line.split(" ")
            label = split[0]
            x0, y0, bw, bh = [int(val) for val in split[1:5]]
            x1 = x0 + bw
            y1 = y0 + bh
            label_int = model_labels[label]
            if label_int != 1:  # skip crowds and invalid numbers
                continue
            elif np.nan in (x0, y0, x1, y1) or np.inf in (x0, y0, x1, y1):
                continue
            elif not (x0 < x1 and y0 < y1):
                continue

            box_data.append([x0 / width, y0 / height, x1 / width, y1 / height, label_int])

    return np.array(box_data)


def prep_data():
    # useful directories
    dataset_dir = "D:/.datasets/CALTECH_PEDESTRIAN/unpacked"
    img_dir = f"{dataset_dir}/images"
    annot_dir = f"{dataset_dir}/annotations"

    # collect data here
    data = []

    # match sets to images
    for set in os.listdir(annot_dir):
        print(set)
        for vdir in os.listdir(f"{annot_dir}/{set}"):
            print(f"  {vdir}")
            for annot_file in os.listdir(f"{annot_dir}/{set}/{vdir}"):
                num = annot_file[1:-4]
                img_name = f"{set}_{vdir}_0{num}.jpg"  # get image name (will have to change once image names are fixed)
                img_path = f"{img_dir}/{set}/{img_name}"  # assemble image path
                annot_path = f"{annot_dir}/{set}/{vdir}/{annot_file}"  # assemble annotation path

                # extract boxes
                boxes = extract_box_caltech(annot_path)

                if len(boxes) == 0: # remove all frames with no boxes
                    continue

                # append to full set
                data.append({
                    "image": img_path,
                    "boxes": boxes
                })
        break
    return data


def caltech(get_pickle=True):
    if get_pickle and exists("pickle/sequence.p"):
        sequence = pickle.load(open("pickle/sequence.p", "rb"))
        return sequence

    data = prep_data()

    class_names = ['person', 'people']
    augmentator = AugmentCaltech(num_classes=len(class_names))

    draw_boxes = SequentialProcessor([
        pr.ControlMap(pr.ToBoxes2D(class_names, True), [1], [1]),
        pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}),
        pr.DrawBoxes2D(class_names),
        pr.ShowImage()])

    print('Image and boxes augmentations with generator')
    batch_size = 5
    sequence = ProcessingSequence(augmentator, batch_size, data)
    for i in sample(range(0, 1000), 5):
        try:
            batch = sequence.__getitem__(i)
        except:
            print("continue")
            continue
        batch_images, batch_boxes = batch[0]['image'], batch[1]['boxes']
        image, boxes = batch_images[0], batch_boxes[0]
        image = deprocess_image(image)
        print(boxes[0])
        for i, b in enumerate(boxes):
            if b[0] >= b[2] or b[1] >= b[3]:
                np.delete(boxes, i)

    return sequence


if __name__ == "__main__":
    caltech()

import os
import random

from pickle import dump

import numpy as np

"""class_labels = {
    "background": 0,
    "person": 1,
    "people": 2
}"""

class_labels = {
    "background": 0,
    "person": 1,
#    "person-fa": 2,
#    "person?": 3,
    "people": 2
}

class_names = list(class_labels.keys())

PICKLE_DIR = "pickle"
DATASET_DIR = "C:/Users/Farrukh/Documents/.data/CALTECH_PEDESTRIAN"

count = {
    "background": 0,
    "person": 0,
    "person-fa": 0,
    "person?": 0,
    "people": 0
}


def extract_box_caltech(file, width=640, height=480):
    """
    Extracts bounding boxes from the provided file.
    :param file: path to annotations file.
    :param width: image width. default is 640px (caltech pedestrian dataset size)
    :param height: image height. default is 480px (caltech pedestrian dataset size)
    :returns list of bounding boxes.
    """
    model_labels = class_labels

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
            try:  # skip invalid labels
                label_int = model_labels[label]
            except KeyError:
                continue

            count[label] += 1
            # skip invalid coordinates
            if np.nan in (x0, y0, x1, y1) or np.inf in (x0, y0, x1, y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue
            elif not (x0 < x1 and y0 < y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue

            box_data.append([x0 / width, y0 / height, x1 / width, y1 / height, label_int])

    return np.array(box_data)


def negative_samples():
    boxes = []

    for _ in range(9): # roughly 1 negative image for each positive box, so add 9 boxes to uphold 3 neg/pos box ratio
        x = [random.uniform(0, 1) for c in range(2)]
        y = [random.uniform(0, 1) for c in range(2)]
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)

        boxes.append([x_min, y_min, x_max, y_max, class_labels['background']])

    return np.array(boxes)


def prep_data(dataset_dir=DATASET_DIR):
    """
    Prepares data by converting to required representations and formatting as a list
    :param dataset_dir: Path to the dataset directory.
    :return: Dictionary with image filepaths and bounding boxes
    """
    # useful directories
    img_dir = f"{dataset_dir}/images"
    annot_dir = f"{dataset_dir}/annotations"

    # collect data here
    data = []

    # match sets to images
    set_count = 0
    for s in os.listdir(annot_dir):
        img_set = []  # initialize set

        print(s)
        for vdir in os.listdir(f"{annot_dir}/{s}"):
            print(f"  {vdir}")
            for annot_file in os.listdir(f"{annot_dir}/{s}/{vdir}"):
                num = annot_file[1:-4]

                # the image/annotation paths are not following the format due to different scripts used for images and
                # annotations extracted using piotr dollar's toolbox
                img_name = f"{s}_{vdir}_0{num}.jpg"  # get image name
                img_path = f"{img_dir}/{s}/{img_name}"  # assemble image path
                annot_path = f"{annot_dir}/{s}/{vdir}/{annot_file}"  # assemble annotation path

                # extract boxes
                boxes = extract_box_caltech(annot_path)

                if len(boxes) == 0 and set_count < 6:  # remove all frames with no boxes
                    boxes = negative_samples()
                elif set_count >= 6 and len(boxes) == 0:
                    continue

                img_set.append({
                    "image": img_path,
                    "boxes": boxes
                })

                set_count += 1

        # append to full set
        data += img_set
        print(count)
        # save set
        dump(img_set, open(f"{PICKLE_DIR}/by_set_AAA/{s}.p", "wb"))

    return data


if __name__ == "__main__":
    data = prep_data(dataset_dir=DATASET_DIR)  # prep dataset
    dump(data, open(f"{PICKLE_DIR}/dataset.p", "wb"))

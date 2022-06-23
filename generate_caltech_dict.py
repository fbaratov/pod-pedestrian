import os
from pickle import dump

import numpy as np

class_labels = {
    "background": 0,
    "person": 1,
    "person-fa": 2,
    "person?": 3,
    "people": 4
}
class_names = list(class_labels.keys())

PICKLE_DIR = "pickle/"
DATASET_DIR = "C:/Users/Farrukh/Documents/.data/CALTECH_PEDESTRIAN"


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
        "person-fa": 2,
        "person?": 3,
        "people": 4
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
            """if label_int != 1:  # skip crowds and invalid numbers
                # print(f"Invalid label: {x0, y0, x1, y1, label}")
                continue"""
            # skip invalid coordinates
            if np.nan in (x0, y0, x1, y1) or np.inf in (x0, y0, x1, y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue
            elif not (x0 < x1 and y0 < y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue

            box_data.append([x0 / width, y0 / height, x1 / width, y1 / height, label_int])

    return np.array(box_data)


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
    for s in os.listdir(annot_dir):
        print(s)
        for vdir in os.listdir(f"{annot_dir}/{s}"):
            print(f"  {vdir}")
            for annot_file in os.listdir(f"{annot_dir}/{s}/{vdir}"):
                num = annot_file[1:-4]

                # the image/annotation paths are not following the format due to different scripts used for images and
                # annotations extracted using piotr dollar's toolbox
                img_name = f"{s}_{vdir}_0{num}.jpg"  # get image name (will have to change once image names are fixed)
                img_path = f"{img_dir}/{s}/{img_name}"  # assemble image path
                annot_path = f"{annot_dir}/{s}/{vdir}/{annot_file}"  # assemble annotation path

                # extract boxes
                boxes = extract_box_caltech(annot_path)

                if len(boxes) == 0:  # remove all frames with no boxes
                    continue

                # append to full set
                data.append({
                    "image": img_path,
                    "boxes": boxes
                })
    return data


if __name__ == "__main__":
    data = prep_data(dataset_dir=DATASET_DIR)  # prep dataset
    dump(data, open(f"{PICKLE_DIR}dataset.p", "wb"))

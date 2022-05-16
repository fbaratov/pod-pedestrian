import os
from os.path import exists
from random import sample

from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import PreprocessImage, AugmentDetection

import pickle

import numpy as np
from paz.abstract import ProcessingSequence, SequentialProcessor
import paz.processors as pr
from sklearn.model_selection import train_test_split

class_labels = {
    "background": 0,
    "person": 1
}
class_names = list(class_labels.keys())


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
        "person-fa": 3,
        "person?": 4,
        "people": 5
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
                # print(f"Invalid label: {x0, y0, x1, y1, label}")
                continue
            elif np.nan in (x0, y0, x1, y1) or np.inf in (x0, y0, x1, y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue
            elif not (x0 < x1 and y0 < y1):
                # print(f"Invalid coords: {x0, y0, x1, y1, label}")
                continue

            box_data.append([x0, y0, x1, y1, label_int])

    return np.array(box_data)


def prep_data(discard_negatives=False):
    """
    Prepares data by converting to required representations and formatting as a list
    :param discard_negatives: If True, does not include images without people in the dataset.
    :return: Dictionary with image filepaths and bounding boxes
    """
    # useful directories
    dataset_dir = "D:/.datasets/CALTECH_PEDESTRIAN/unpacked"
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
                img_name = f"{s}_{vdir}_0{num}.jpg"  # get image name (will have to change once image names are fixed)
                img_path = f"{img_dir}/{s}/{img_name}"  # assemble image path
                annot_path = f"{annot_dir}/{s}/{vdir}/{annot_file}"  # assemble annotation path

                # extract boxes
                boxes = extract_box_caltech(annot_path)

                if len(boxes) == 0:  # remove all frames with no boxes
                    boxes.append(np.array([0, 1, 0, 1, 0]))

                # append to full set
                data.append({
                    "image": img_path,
                    "boxes": boxes
                })
    return data


def load_data(use_saved, test_split, val_split, train_subset):
    """
    Loads train/val/test splits
    :param use_saved: If True, uses preexisting splits. Otherwise creates new ones.
    :param test_split: Size of test split
    :param val_split: Size of validation split
    :param train_subset: Size of subset of training split that will be used (used to lower training time)
    :return: Train, validation, and test splits as lists of dictionaries
    """
    if not exists("pickle/dataset.p"):
        data = prep_data()
        pickle.dump(data, open("pickle/dataset.p", "wb"))
        train_data, test_data = train_test_split(data, test_size=test_split)
        pickle.dump(train_data, open("pickle/train.p", "wb"))
        pickle.dump(test_data, open("pickle/test.p", "wb"))

    if use_saved and exists("pickle/train.p") and exists("pickle/test.p"):
        train_data = pickle.load(open("pickle/train.p", "rb"))
        test_data = pickle.load(open("pickle/test.p", "rb"))
    else:
        data = pickle.load(open("pickle/dataset.p", "rb"))
        # split d_test/d_train
        train_data, test_data = train_test_split(data, test_size=test_split)
        pickle.dump(train_data, open("pickle/train.p", "wb"))
        pickle.dump(test_data, open("pickle/test.p", "wb"))

    # get train data subset
    if train_subset:
        train_data = sample(train_data, k=int(train_subset * len(train_data)))

    # get validation subset
    train_data, val_data = train_test_split(train_data, test_size=val_split)

    return train_data, val_data, test_data


def caltech(use_saved=True, train_subset=None, test_split=0.3, val_split=0.1, batch_size=16):
    """
    Creates a processor that can be used to d_train a model on the caltech pedestrian dataset.
    :param train_subset: How much of the training subset to use for the training dataset represented as a decimal if None, uses whole subset.
    :param test_split: How much of the whole dataset will be used for testing.
    :param val_split: How much of the training dataset will be used for validation.
    :param use_saved: If True, uses saved splits instead of generating new ones.
    :param batch_size: Input batch size.
    :return: Train and d_test processors for the data.
    """
    train_data, val_data, test_data = load_data(use_saved, test_split, val_split, train_subset)

    # create processor
    prior_boxes = create_prior_boxes('VOC')
    processor = AugmentDetection(prior_boxes, split=pr.TEST, num_classes=len(class_names), size=300,
                                 mean=None, IOU=.5,
                                 variances=[0.1, 0.1, 0.2, 0.2])

    # create and save sequences
    train_seq = ProcessingSequence(processor, batch_size, train_data)
    val_seq = ProcessingSequence(processor, batch_size, val_data)

    print(f"train size: {len(train_data)}\ntest size: {len(test_data)}\nvalidation size: {len(val_data)}")
    return train_seq, val_seq, test_data


if __name__ == "__main__":
    caltech()

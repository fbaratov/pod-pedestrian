import os
from os.path import exists

from paz.pipelines import PreprocessImage

import pickle

import numpy as np
from paz.abstract import ProcessingSequence, SequentialProcessor
from paz.models.detection.utils import create_prior_boxes
import paz.processors as pr
from sklearn.model_selection import train_test_split


class CaltechPreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes. Modified version of PreprocessBoxes class.

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """

    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(CaltechPreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU), )
        # self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentCaltech(SequentialProcessor):
    """Modified version of AugmentProcessor class.

    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """

    def __init__(self, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentCaltech, self).__init__()
        # image processors
        # self.augment_image = AugmentImage()
        # self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        # self.augment_boxes = AugmentBoxes()
        prior_boxes = create_prior_boxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = CaltechPreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        # if split == pr.TRAIN:
        #   self.add(pr.ControlMap(self.augment_image, [0], [0]))
        #   self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


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


def prep_data(test_split=0.3):
    """
    Prepares data by converting to required representations and formatting as a list
    """
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

                if len(boxes) == 0:  # remove all frames with no boxes
                    continue

                # append to full set
                data.append({
                    "image": img_path,
                    "boxes": boxes
                })
    return data


def caltech(get_pickle=True):
    """
    Creates a processor that can be used to d_train a model on the caltech pedestrian dataset.
    :param get_pickle: If True, uses saved splits instead of generating new ones.
    :return: Train and d_test processors for the data.
    """
    if get_pickle and exists("pickle/train.p") and exists("pickle/test.p"):
        train_data = pickle.load(open("pickle/train.p", "rb"))
        test_data = pickle.load(open("pickle/test.p", "rb"))
    else:
        data = prep_data()
        # split d_test/d_train
        train_data, test_data = train_test_split(data, test_size=0.2)
        pickle.dump(train_data, open("pickle/train.p", "wb"))
        pickle.dump(test_data, open("pickle/test.p", "wb"))
    print(f"train size: {len(train_data)}\ntest size: {len(test_data)}")
    # create classes/augmentator
    class_names = ['person', 'people']
    augmentator = AugmentCaltech(num_classes=len(class_names))

    # create and save sequences
    batch_size = 5
    train_seq = ProcessingSequence(augmentator, batch_size, train_data)
    return train_seq, test_data


if __name__ == "__main__":
    caltech()

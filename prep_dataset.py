from os.path import exists
from random import sample

from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentDetection

import pickle
from paz.abstract import ProcessingSequence
import paz.processors as pr
from sklearn.model_selection import train_test_split
from generate_caltech_dict import class_labels, class_names, PICKLE_DIR


def load_data(use_saved, test_split, val_split, subset):
    """
    Loads train/val/test splits from pickles
    :param use_saved: If True, uses preexisting splits. Otherwise creates new ones.
    :param test_split: Size of test split
    :param val_split: Size of validation split
    :param subset: How much of each split to use (used to lower training time)
    :return: Train, validation, and test splits as lists of dictionaries
    """
    if not exists(f"{PICKLE_DIR}/dataset.p"):  # no dataset file available
        raise FileNotFoundError("No dataset dictionary found! Please use generate_caltech_dict.py to create one.")
    if use_saved and exists(f"{PICKLE_DIR}/train.p") and exists(f"{PICKLE_DIR}/test.p"):
        train_data = pickle.load(open(f"{PICKLE_DIR}/train.p", "rb"))
        test_data = pickle.load(open(f"{PICKLE_DIR}/test.p", "rb"))
    else:
        data = pickle.load(open(f"{PICKLE_DIR}/dataset.p", "rb"))
        # split d_test/d_train
        train_data, test_data = train_test_split(data, test_size=test_split + val_split)
        pickle.dump(train_data, open(f"{PICKLE_DIR}/train.p", "wb"))
        pickle.dump(test_data, open(f"{PICKLE_DIR}/test.p", "wb"))

    # get validation subset
    test_data, val_data = train_test_split(test_data, test_size=val_split / (val_split + test_split))

    # get train data subset
    if subset:
        train_data = sample(train_data, k=int(subset * len(train_data)))
        test_data = sample(test_data, k=int(subset * len(test_data)))
        val_data = sample(val_data, k=int(subset * len(val_data)))

    return train_data, val_data, test_data


def caltech(use_saved=True, subset=None, test_split=0.3, val_split=0.1, batch_size=16):
    """
    Creates a processor from a filepath/bbox dictionary that can be used to train a model.
    :param subset: How much of each split to use. If None, uses whole split.
    :param test_split: Size of test split
    :param val_split: Size of validation split
    :param use_saved: If True, uses saved splits instead of generating new ones.
    :param batch_size: Input batch size.
    :return: Train and d_test processors for the data.
    """
    train_data, val_data, test_data = load_data(use_saved, test_split, val_split, subset)

    # create processor
    prior_boxes = create_prior_boxes('VOC')
    train_processor = AugmentDetection(prior_boxes, split=pr.TRAIN, num_classes=len(class_names), size=300,
                                       mean=None, IOU=.5,
                                       variances=[0.1, 0.1, 0.2, 0.2])
    val_processor = AugmentDetection(prior_boxes, split=pr.VAL, num_classes=len(class_names), size=300,
                                     mean=None, IOU=.5,
                                     variances=[0.1, 0.1, 0.2, 0.2])

    # create and save sequences
    train_seq = ProcessingSequence(train_processor, batch_size, train_data)
    val_seq = ProcessingSequence(val_processor, batch_size, val_data)

    print(f"train size: {len(train_data)}\ntest size: {len(test_data)}\nvalidation size: {len(val_data)}")
    return train_seq, val_seq, test_data


if __name__ == "__main__":
    # adding comment to tests push access
    caltech()

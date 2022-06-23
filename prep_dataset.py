from os import mkdir
from os.path import exists, isdir
from random import sample

from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentDetection

import pickle
from paz.abstract import ProcessingSequence
import paz.processors as pr
from sklearn.model_selection import train_test_split
from generate_caltech_dict import class_labels, class_names, PICKLE_DIR


def load_data(split_name, test_size=0.3, val_size=.1, subset=None):
    """
    Loads train/val/test splits from pickles
    :param split_name: Name of split set.
    :param test_size: Size of test split
    :param val_size: Size of validation split
    :param subset: How much of each split to use (used to lower training time)
    :return: Train, validation, and test splits as lists of dictionaries
    """
    if not exists(f"{PICKLE_DIR}/dataset.p"):  # no dataset file available
        raise FileNotFoundError("No dataset dictionary found! Please use generate_caltech_dict.py to create one.")
    elif not exists(f"{PICKLE_DIR}/{split_name}"):
        print(f"No split {split_name} found, generating new one...")
        create_splits(split_name, test_size, val_size, subset)

    train_data = pickle.load(open(f"{PICKLE_DIR}/{split_name}/train.p", "rb"))
    test_data = pickle.load(open(f"{PICKLE_DIR}/{split_name}/test.p", "rb"))
    val_data = pickle.load(open(f"{PICKLE_DIR}/{split_name}/validation.p", "rb"))

    return train_data, val_data, test_data


def create_splits(split_name=None, test_size=0.15, val_size=.15, subset=None):
    """
    Create data splits.
    :param test_size:
    :param val_size:
    :param subset:
    :param split_name:
    :return:
    """
    if split_name is None:
        i = 0
        while isdir(f"{PICKLE_DIR}/split{i}"):
            i += 1
        split_name = f"split{i}"

    split_dir = f"{PICKLE_DIR}/{split_name}"

    if not isdir(split_dir):
        mkdir(split_dir)

    data = pickle.load(open(f"{PICKLE_DIR}/dataset.p", "rb"))

    # split d_test/d_train
    train_data, test_data = train_test_split(data, test_size=test_size + val_size)

    # get validation subset
    test_data, val_data = train_test_split(test_data, test_size=val_size / (val_size + test_size))

    # get train data subset
    if subset:
        train_data = sample(train_data, k=int(subset * len(train_data)))
        test_data = sample(test_data, k=int(subset * len(test_data)))
        val_data = sample(val_data, k=int(subset * len(val_data)))

    # save splits
    pickle.dump(train_data, open(f"{split_dir}/train.p", "wb"))
    pickle.dump(test_data, open(f"{split_dir}/test.p", "wb"))
    pickle.dump(val_data, open(f"{split_dir}/validation.p", "wb"))


def retrieve_splits(split_name, subset=None, test_split=0.15, val_split=0.15, batch_size=16):
    """
    Creates a processor from a filepath/bbox dictionary that can be used to train a model.
    :param split_name: Name of dataset splits to retrieve.
    :param subset: How much of each split to use. If None, uses whole split.
    :param test_split: Size of test split
    :param val_split: Size of validation split
    :param batch_size: Input batch size.
    :return: Train and d_test processors for the data.
    """
    train_data, val_data, test_data = load_data(split_name, test_split, val_split, subset)

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
    load_data(split_name="all_classes_70_15_15")

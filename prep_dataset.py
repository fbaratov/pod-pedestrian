from os import mkdir
from os.path import exists, isdir

from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentDetection

import pickle
from paz.abstract import ProcessingSequence
import paz.processors as pr
from sklearn.model_selection import train_test_split
from generate_caltech_dict import class_labels, class_names, PICKLE_DIR


def load_data(split_dir):
    """
    Loads train/val/test splits from pickles
    :param split_dir: Directory of split set.
    :return: Train, validation, and test splits as lists of dictionaries
    """
    if not exists(f"{split_dir}"):
        raise FileNotFoundError(f"No split {split_dir} found!")

    train_data = pickle.load(open(f"{split_dir}/train.p", "rb"))
    test_data = pickle.load(open(f"{split_dir}/test.p", "rb"))
    val_data = pickle.load(open(f"{split_dir}/validation.p", "rb"))

    return train_data, val_data, test_data


def create_caltech_splits(split_dir, caltech_dir):
    """
    Create data splits based on caltech sets.
    :param split_dir: Directory to save splits in
    :param caltech_dir: Directory with caltech data sets.
    :return:
    """
    if not isdir(split_dir):
        mkdir(split_dir)

    train_sets = ["set00.p", "set01.p", "set02.p", "set03.p", "set04.p", "set05.p"]
    test_sets = ["set06.p", "set07.p", "set08.p", "set09.p"]
    val_sets = ["set10.p"]

    train_data, test_data, val_data = [], [], []

    for fname in train_sets:
        tset = pickle.load(open(f"{caltech_dir}/{fname}", "rb"))
        train_data += tset

    for fname in test_sets:
        tset = pickle.load(open(f"{caltech_dir}/{fname}", "rb"))
        test_data += tset

    for fname in val_sets:
        vset = pickle.load(open(f"{caltech_dir}/{fname}", "rb"))
        val_data += vset

    pickle.dump(train_data, open(f"{split_dir}/train.p", "wb"))
    pickle.dump(test_data, open(f"{split_dir}/test.p", "wb"))
    pickle.dump(val_data, open(f"{split_dir}/validation.p", "wb"))


def create_splits(dataset_path, split_dir, test_size=0.15, val_size=.15):
    """
    Create data splits.
    :param dataset_path:
    :param split_dir:
    :param test_size:
    :param val_size:
    :return:
    """

    if not isdir(split_dir):
        mkdir(split_dir)

    data = pickle.load(open(f"{dataset_path}.p", "rb"))

    # split test/train
    train_data, test_data = train_test_split(data, test_size=test_size + val_size)

    # get validation subset
    test_data, val_data = train_test_split(test_data, test_size=val_size / (val_size + test_size))

    # save splits
    pickle.dump(train_data, open(f"{split_dir}/train.p", "wb"))
    pickle.dump(test_data, open(f"{split_dir}/test.p", "wb"))
    pickle.dump(val_data, open(f"{split_dir}/validation.p", "wb"))


def retrieve_splits(split_name, batch_size=16):
    """
    Creates a processor from a filepath/bbox dictionary that can be used to train a model.
    :param split_name: Name of dataset splits to retrieve.
    :param batch_size: Input batch size.
    :return: Train and validation splits as processors, test split as dictionary .
    """
    train_data, val_data, test_data = load_data(split_name)

    # create processor
    prior_boxes = create_prior_boxes('VOC')

    train_processor = AugmentDetection(prior_boxes, split=pr.VAL, num_classes=len(class_names), size=300,
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
    create_caltech_splits("pickle/caltech_split", "pickle/by_set_")
    retrieve_splits("pickle/caltech_split")

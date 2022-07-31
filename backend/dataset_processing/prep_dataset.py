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
    """ Loads train/val/test splits from pickles

    Args:
        split_dir: Directory of split set.

    Returns
        Train, validation, and test splits as lists of dictionaries containing keys 'image' and 'boxes'.
    """
    if not exists(f"{split_dir}"):
        raise FileNotFoundError(f"No split {split_dir} found!")

    train_data = pickle.load(open(f"{split_dir}/train.p", "rb"))
    test_data = pickle.load(open(f"{split_dir}/test.p", "rb"))
    val_data = pickle.load(open(f"{split_dir}/validation.p", "rb"))

    return train_data, val_data, test_data


def create_splits(dataset_path, split_dir, test_size=0.15, val_size=.15):
    """ Create data splits by shuffling dataset into train,test,validation sets.

    Args:
        dataset_path: Path to pickle containing list of dictionaries with keys 'images' and 'boxes'.
        split_dir: Directory in which to save dataset splits.
        test_size: Size of test split. Float between [0,1].
        val_size: Size of validation split. Float between [0,1].
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
    """Creates a processor from a filepath/bbox dictionary that can be used to train a model.

    Args:
        split_name: Directory in which splits are located. String.
        batch_size: Input batch size. Int.

    Returns
        Train and validation splits as PAZ processors, test split as dictionary with keys 'images' and 'boxes.
    """
    train_data, val_data, test_data = load_data(split_name)

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
    retrieve_splits("pickle/all_classes_70_15_15")

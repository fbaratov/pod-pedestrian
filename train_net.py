from keras.layers import Reshape
from keras.models import load_model
from os.path import exists

from custom_map import evaluateMAP
from paz.models.detection.ssd300 import SSD300
from paz.optimization.callbacks import EvaluateMAP

from CaltechLoader import DictLoader
from prep_caltech import caltech


def fit_model(model, data, callbacks=None):
    """
    Fits model to data.
    :param model: Model to fit
    :param data: Training dataset
    :return: Model fitted to data
    """
    if callbacks is None:
        callbacks = []
    model.fit(data, callbacks=callbacks)
    model.save("models/model")
    return model


def get_model(use_saved=False):
    """
    Creates and compiles a SSD300 model.
    :param use_saved: If True, uses saved model instead of training a new one.
    :return: SSD300 model
    """
    if use_saved and exists("models/model"):
        model = load_model("models/model")
    else:
        model = SSD300(num_classes=2, base_weights=None, head_weights=None)
        model.compile()

    return model


def get_splits(use_saved=False):
    """
    Gets train/test splits of the data.
    :param use_saved: If True, uses saved splits instead of making new ones.
    :return: Processors for train/test splits
    """
    train, test = caltech(use_saved)
    return train, test


def train_model(data, saved_model=False):
    """
    Trains model on given data or retrieves trained model.
    :param data: Training data as a processing sequence
    :param saved_model: If True, uses saved model instead of training new one.
    :return: SSD300 model
    """
    model = get_model(saved_model)
    if not saved_model:
        model = fit_model(model, data)
    return model


def evaluate_model(test, model):
    """
    Evaluates model using given dataset.
    :param test: Test dataset as a dict of format {img_path: [bbox_vectors]}
    :param model: PAZ model
    :return:
    """
    test_data = test
    score = evaluateMAP(model, test_data, {"person": 1, "people": 2})
    print(score)
    # print('Test loss:', score[0])
    # ('Test accuracy:', score[1])


if __name__ == "__main__":
    saved_data = True
    saved_model = True and saved_data

    train, test = get_splits(saved_data)
    model = train_model(train, saved_model)
    evaluate_model(test, model)

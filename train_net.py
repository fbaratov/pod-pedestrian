from keras.models import load_model
from os.path import exists

from paz.models.detection.ssd300 import SSD300

from prep_caltech import caltech


def fit_model(model, data):
    """
    Fits model to data.
    :param model: Model to fit
    :param data: Training dataset
    :return: Model fitted to data
    """
    model.fit(data)
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


def train_model(saved_data=False, saved_model=False):
    train, _ = get_splits(saved_data)
    model = get_model(saved_model)
    if not saved_model:
        model = fit_model(model, train)
    return model


def evaluate_model(model):
    _, test = get_splits(True)
    score = model.evaluate(test, verbose=0)
    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)

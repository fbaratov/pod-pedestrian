from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from paz.models import SSD300

from prep_dataset import retrieve_splits
from ssd_baseline import SSD300_baseline
from trainer import *
from generate_caltech_dict import class_labels, class_names


def train_net(epochs=10, rate=0.3):
    """
    Takes care of training, evaluating, and displaying network results.
    :param epochs: Number of epochs to train for.
    :param rate: Model dropout rate.
    """

    model = init_ssd(rate=0.3)

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=5,
                        min_delta=0.005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    # train model and plot loss
    hist = train_model(model, d_train, d_val, save_dir, callbacks=cb, epochs=epochs)


if __name__ == "__main__":
    train_net(epochs=10, rate=0.3)

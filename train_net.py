from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from prep_dataset import retrieve_splits
from trainer import *


def train_net(epochs=10):
    """
    Takes care of training, evaluating, and displaying network results.
    :param epochs: Number of epochs to train for.
    """

    # create trainer (used to train model/predict/evaluate as well as to create dataset splits)
    split_names = "full_set"
    trainer = Trainer(splits=retrieve_splits(split_names))

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=5,
                        min_delta=0.005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    # train model and plot loss
    hist = trainer.train(callbacks=cb, epochs=epochs)

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_net(epochs=5)

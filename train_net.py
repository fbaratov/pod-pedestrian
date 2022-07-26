from keras.callbacks import EarlyStopping, ModelCheckpoint

from prep_dataset import retrieve_splits
from trainer import *
from generate_caltech_dict import class_labels, class_names


def train_net(model, split_dir, save_dir, epochs=10):
    """
    Takes care of training, evaluating, and displaying network results.
    :param save_dir:
    :param model: Model directory
    :param split_dir: Directory from which to retrieve data splits.
    :param epochs: Number of epochs to train for.
    """

    d_train, d_val, _ = retrieve_splits(split_dir)

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=5,
                        min_delta=0.2,
                        verbose=1,
                        restore_best_weights=True),
          ModelCheckpoint(f"models/checkpoint/model.h5", save_weights_only=False),
          ]


    # train model and plot loss, save figure
    hist = train_model(model, d_train, d_val, save_dir, callbacks=cb, epochs=epochs)


if __name__ == "__main__":
    model = init_ssd(0.3)
    split_dir = "pickle/caltech_split"
    save_dir = "models/caltech_model_1"

    train_net(model, split_dir, save_dir, epochs=10)

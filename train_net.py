from keras.callbacks import EarlyStopping

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
                        patience=15,
                        min_delta=0.02,
                        verbose=1,
                        restore_best_weights=True),
          ]


    # train model and plot loss, save figure
    hist = train_model(model, d_train, d_val, save_dir, callbacks=cb, epochs=epochs)

if __name__ == "__main__":
    model = init_ssd(0.3)
    split_dir = "pickle/all_classes_70_15_15"
    save_dir = "models/model_1"

    train_net(model, split_dir, save_dir, epochs=30)

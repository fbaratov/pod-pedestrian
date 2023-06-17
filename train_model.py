from keras.callbacks import EarlyStopping
import argparse
from backend.dataset_processing.prep_dataset import retrieve_splits
from backend.model_utils import *


def train_net(model, split_dir, save_dir, epochs=10):
    """
    Takes care of training, evaluating, and displaying network results.

    Args:
        save_dir: Directory to save model in.
        model: PAZ SSD model.
        split_dir: Directory where data splits are located.
        epochs: Number of epochs to train for. Int.
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
    args = argparse.ArgumentParser(
                    prog='Dropout SSD300 trainer',
                    description='Trains the SSD300')
    
    args.add_argument('--split_dir', help='directory containing train/test/validation splits.')
    args.add_argument('--save_dir', help="directory to which the model should be saved.")
    args.add_argument('--dropout_rate', help="model dropout rate. Set to 0. for no dropout.", default=0.3)
    args.add_argument('--epochs', help='number of epochs for which the model should be trained.', default=30)

    model = init_ssd(args.dropout_rate)

    train_net(model, args.split_dir, args.save_dir, epochs=args.epochs)

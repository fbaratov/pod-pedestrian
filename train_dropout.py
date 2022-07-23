from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from matplotlib import pyplot as plt

from generate_caltech_dict import class_labels, class_names
from prep_dataset import retrieve_splits
from ssd_dropout import SSD300_dropout
from trainer import DropoutTrainer


def model_fn(prob=0.3):
    ssd = SSD300_dropout(num_classes=len(class_names), base_weights='VGG', head_weights=None, prob=prob)
    return ssd


def train_dropout(split_name, model_name, prob):
    # create new dropout model
    model = model_fn(prob)

    # initialize trainer with model and set name
    trainer = DropoutTrainer(splits=retrieve_splits(split_name),
                             model=model)
    trainer.model_name = model_name

    # plot model structure for review
    plot_model(
        trainer.model,
        to_file=f"{trainer.model_name}.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
          patience=5,
          min_delta=0.005,
          verbose=1,
          restore_best_weights=True)
          ]

    # train model and plot loss
    hist = trainer.train(callbacks=cb, epochs=10)

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_dropout(split_name="caltech_split",
                  model_name="dropout_model_caltech_0",
                  prob=0.3)

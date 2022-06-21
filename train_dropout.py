from keras import Model
from keras.utils import plot_model

from ssd_dropout import SSD300_dropout
from train_net import *


def model_fn(prob=0.3):
    ssd = SSD300_dropout(num_classes=len(class_names), base_weights='VGG', head_weights=None, prob=prob)
    return ssd


if __name__ == "__main__":
    model = model_fn(0.3)

    trainer = DropoutTrainer(saved_data=True,
                             model_name=None,
                             subset=1,
                             test_split=0.15,
                             val_split=0.15,
                             batch_size=8
                             )
    trainer.init_model(model, "model_dropout_bigboi")

    model = trainer.model

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=5,
                        min_delta=0.005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    # train model and plot loss
    hist = trainer.train(callbacks=cb, epochs=10)

    # convert model to two-headed model
    trainer.init_model(Model(model.input, [model.layers[-3].output, model.layers[-2].output]))

    trainer.is_trained = True

    plot_model(
        trainer.model,
        to_file="circumcised_model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )

    if hist:
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.legend()
        plt.show()

    trainer.show_results(k=100, show_truths=True, score_thresh=.1, nms=.4)

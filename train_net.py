from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from trainer import *


def pipeline(saved_data=True, model_name="model", subset=1., epochs=10):
    """
    Takes care of training, evaluating, and displaying network results.
    :param saved_data: If False, generates new data splits, otherwise uses saved.
    :param model_name: Name of model to retrieve/train
    :param subset: Portion of dataset to use. 1 uses the whole dataset. Used to reduce training time.
    :param epochs: Number of epochs to train for.
    """

    # create trainer (used to train model/predict/evaluate as well as to create dataset splits)
    trainer = Trainer(saved_data,
                      model_name=model_name,
                      subset=subset,
                      test_split=0.15,
                      val_split=0.15,
                      batch_size=16
                      )

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=5,
                        min_delta=0.005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    # train model and plot loss
    hist = trainer.train(callbacks=cb, epochs=epochs)

    if hist:
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.legend()
        plt.show()

    print(trainer.evaluate())

    if hist:
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.legend()
        plt.show()

    # draw results
    trainer.show_results(k=100, show_truths=True)


if __name__ == "__main__":
    pipeline(saved_data=True, model_name="model", subset=.05, epochs=5)

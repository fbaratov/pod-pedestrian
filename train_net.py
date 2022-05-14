from random import sample

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from paz.backend.image import load_image, convert_color_space, BGR2RGB
from paz.processors import ShowImage, BGR_IMAGENET_MEAN

import pickle

from keras.models import load_model
from os.path import exists

from paz.evaluation import evaluateMAP
from paz.models.detection.ssd300 import SSD300
from paz.pipelines.detection import DetectSingleShot

from prep_caltech import caltech


class Trainer:
    """
    Trains model to data.
    """

    def __init__(self, saved_data=False, saved_model=False, train_subset=0.1, test_split=0.3, val_split=0.1,
                 batch_size=16):
        """
        Initializes model and train/test splits.
        :param saved_data: If True, uses saved dataset splits instead of making new ones.
        :param saved_model: If True, uses saved model instead of training a new one.
        :param train_subset: decimal representing which portion of a subset to use.
        :param batch_size: Batch size for training/val
        """

        self.d_train, self.d_val, self.d_test, self.model = None, None, None, None

        self.is_trained = saved_model and saved_data  # if new splits are generated, model needs retraining
        self.get_model(saved_model)
        self.get_splits(saved_data, train_subset, test_split, val_split, batch_size)

    def get_model(self, use_saved):
        """
        Creates and compiles a SSD300 model.
        :param use_saved: If True, uses saved model instead of training a new one.
        :return: SSD300 model
        """
        if use_saved and exists("models/model"):
            self.model = load_model("models/model")
            self.model.prior_boxes = pickle.load(open("models/prior_boxes.p", "rb"))
        else:
            self.model = SSD300(num_classes=2, base_weights=None, head_weights=None)
            self.model.compile()

    def get_splits(self, use_saved, train_subset, test_split, val_split, batch_size):
        """
        Gets d_train/d_test splits of the data.
        :param train_subset: How much of the training subset to use.
        :param batch_size: Training/validation batch sizes
        :param test_split: How much of dataset goes towards testing
        :param val_split: How much of training dataset goes towards validation.
        :param use_saved: If True, uses saved splits instead of making new ones.
        :return: Processors for train/test splits
        """
        self.d_train, self.d_val, self.d_test = caltech(use_saved, train_subset, test_split, val_split, batch_size)

    def train(self, callbacks=None, epochs=10, force_train=False):
        """
        Trains model on given data or retrieves trained model.
        callbacks: List of callbacks to use
        epochs: Number of epochs to train for
        force_train: If True, trains even if model is already trained.
        """
        if callbacks is None:
            callbacks = []

        if not self.is_trained or force_train:
            """            x_val, y_val = [], []
            for x, y in self.d_val:
                x_val.append(x)
                y_val.append(y)"""
            history = self.model.fit(self.d_train, callbacks=callbacks, epochs=epochs, validation_data=self.d_val)

            self.model.save("models/model")
            pickle.dump(self.model.prior_boxes, open("models/prior_boxes.p", "wb"))
            return history
        else:
            print("Model is already trained!")
            return None

    def predict_model(self, img, fp=True):
        """
        Uses model to make a prediction with the given image.
        :param img: Image or image filepath
        :param fp: Set to True if img is a filepath, otherwise False.
        :return: BBox prediction
        """
        image = load_image(img) if fp else img
        detector = DetectSingleShot(self.model, ["person", "people"], .5, .5, draw=True)
        results = detector(image)
        return results

    def evaluate(self):
        """
        Evaluates model using test dataset.
        :return: Model scores or something
        """
        detector = DetectSingleShot(self.model, ["person", "people"], .5, .5, draw=True)
        eval = evaluateMAP(detector, self.d_test, {"person": 1})
        return eval


if __name__ == "__main__":
    saved_data = True
    saved_model = True
    trainer = Trainer(saved_data,
                      saved_model,
                      train_subset=0.1,
                      test_split=0.1,
                      val_split=0.1,
                      batch_size=16
                      )

    cb = [EarlyStopping(monitor='val_loss',
                        patience=1,
                        min_delta=0.0005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    hist = trainer.train(callbacks=cb, epochs=10)
    if hist:
        plt.plot(hist.history["loss"])
        plt.show()

    draw_boxes = ShowImage()
    for d in trainer.d_test:
        fp = d["image"]
        results = trainer.predict_model(fp)
        if not results["boxes2D"]:
            continue
        print(results["boxes2D"])
        draw_boxes(results["image"])
    # print(trainer.evaluate())

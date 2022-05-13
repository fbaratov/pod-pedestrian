from random import random, sample

import numpy as np
from paz.backend.image import load_image
from paz.processors import ShowImage

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

    def __init__(self, saved_data=False, saved_model=False):
        """
        Initializes model and train/test splits.
        :param saved_data: If True, uses saved dataset splits instead of making new ones.
        :param saved_model: If True, uses saved model instead of training a new one.
        """

        self.d_train, self.d_test, self.model = None, None, None

        self.is_trained = saved_model and saved_data  # if new splits are generated, model needs retraining
        self.get_model(saved_model)
        self.get_splits(saved_data)

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

    def get_splits(self, use_saved):
        """
        Gets d_train/d_test splits of the data.
        :param use_saved: If True, uses saved splits instead of making new ones.
        :return: Processors for train/test splits
        """
        self.d_train, self.d_test = caltech(use_saved)

    def train(self, callbacks=None):
        """
        Trains model on given data or retrieves trained model.
        :return: SSD300 model
        """
        if callbacks is None:
            callbacks = []

        if not self.is_trained:
            self.model.fit(self.d_train, callbacks=callbacks)
            self.model.save("models/model")
            pickle.dump(self.model.prior_boxes, open("models/prior_boxes.p", "wb"))
        else:
            print("Model is already trained!")

    def predict_model(self, fp):
        """
        Uses model to make a prediction with the given image.
        :param fp: image filepath
        :return: BBox prediction
        """
        images = load_image(fp)
        detector = DetectSingleShot(self.model, ["person", "people"], .5, .5, draw=True)
        results = detector(images)
        return results

    def evaluate(self):
        """
        Evaluates model using test dataset.
        :return: Model scores or something
        """
        detector = DetectSingleShot(self.model, ["person", "people"], .5, .5, draw=True)
        eval = evaluateMAP(detector, self.d_test, {"person": 1, "people": 2})
        return eval


if __name__ == "__main__":
    saved_data = True
    saved_model = False
    trainer = Trainer(saved_data, saved_model)
    trainer.train()
    draw_boxes = ShowImage()
    for d in sample(trainer.d_test, 10):
        fp = d["image"]
        results = trainer.predict_model(fp)
        print(results)
        draw_boxes(results["image"])
    #print(trainer.evaluate())

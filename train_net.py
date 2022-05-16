from random import sample

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from paz.backend.image import load_image
from paz.processors import ShowImage

import pickle

from keras.models import load_model
from os.path import exists

from paz.evaluation import evaluateMAP
from paz.models.detection.ssd300 import SSD300
from paz.pipelines.detection import DetectSingleShot

from prep_caltech import caltech

class_labels = {
    "background": 0,
    "person": 1
}
class_names = list(class_labels.keys())


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
        self.get_model(saved_model) # get preexisting model/create new one
        self.get_splits(saved_data, train_subset, test_split, val_split, batch_size)

    def get_model(self, use_saved):
        """
        Creates and compiles a SSD300 model.
        :param use_saved: If True, uses saved model instead of training a new one.
        :return: SSD300 model
        """

        if use_saved and exists("models/model"): # load previous model
            self.model = load_model("models/model")
            self.model.prior_boxes = pickle.load(open("models/prior_boxes.p", "rb"))
        else: # create new model
            self.model = SSD300(num_classes=len(class_names), base_weights=None, head_weights=None)
            self.model.compile(optimizer="adam", loss="binary_crossentropy")

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

        # check for callbacks
        if callbacks is None:
            callbacks = []

        # fit the model to test data
        if not self.is_trained or force_train:
            history = self.model.fit(self.d_train, callbacks=callbacks, epochs=epochs, validation_data=self.d_val)
            self.model.save("models/model")
            self.is_trained = True
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
        detector = DetectSingleShot(self.model, class_names, .8, .8, draw=True)
        results = detector(image)
        return results

    def evaluate(self):
        """
        Evaluates model using test dataset.
        :return: Model scores or something
        """
        detector = DetectSingleShot(self.model, class_names, .5, .5, draw=True)
        eval = evaluateMAP(detector, self.d_test, class_labels)
        return eval


if __name__ == "__main__":
    # config parameters (used to skip creating dataset splits/training new model)
    saved_data = False
    saved_model = False

    # create trainer (used to train model/predict/evaluate as well as to create dataset splits)
    trainer = Trainer(saved_data,
                      saved_model,
                      train_subset=0.1,
                      test_split=0.1,
                      val_split=0.1,
                      batch_size=16
                      )

    # callbacks (passed to trainer.train)
    cb = [EarlyStopping(monitor='val_loss',
                        patience=1,
                        min_delta=0.0005,
                        verbose=1,
                        restore_best_weights=True)
          ]

    # train model and plot loss
    hist = trainer.train(callbacks=cb, epochs=10)
    if hist:
        plt.plot(hist.history["loss"])
        plt.show()

    # visualize all images that have bounding boxes
    draw_boxes = ShowImage()
    for i, d in enumerate(trainer.d_test):
        if not i % 500:
            print(f"{i}/{len(trainer.d_test)}")
        fp = d["image"]
        results = trainer.predict_model(fp)
        if not results["boxes2D"]:
            continue

        #print(results["boxes2D"])
        draw_boxes(results["image"])

    # print(trainer.evaluate())

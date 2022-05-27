from random import sample

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from paz.backend.image import load_image
from paz.optimization import MultiBoxLoss
from paz.processors import ShowImage

import pickle
from keras.models import load_model
from os.path import exists, isdir

from keras.optimizer_v2.gradient_descent import SGD
from paz.evaluation import evaluateMAP
from paz.models.detection.ssd300 import SSD300
from paz.pipelines.detection import DetectSingleShot
from prep_caltech import caltech

"""    "class1": 2,
    "class2": 3,
    "class4": 4,
    "class5": 5,
    "class6": 6,
    "class7": 7,
    "class8": 8,
    "class9": 9,
    "class10": 10,
    "class11": 11,
    "class12": 12,
    "class13": 13,
    "class14": 14,
    "class15": 15,
    "class16": 16,
    "class17": 17,
    "class18": 18,
    "class19": 19,
    "class20": 20,"""

class_labels = {
    "background": 0,
    "person": 1,
}
class_names = list(class_labels.keys())


class Trainer:
    """
    Trains model to data.
    """

    def __init__(self, saved_data=True, model_name=None, subset=0.1, test_split=0.3, val_split=0.1,
                 batch_size=16):
        """
        Initializes model and train/test splits.
        :param saved_data: If True, uses saved dataset splits instead of making new ones.
        :param model_name: Save name of model. Existing name if using saved model, otherwise None or custom name.
        :param subset: How much of each split to use (used to lower number of images/training time
        :param batch_size: Batch size for training/val
        """
        self.d_train, self.d_val, self.d_test, self.model = None, None, None, None
        self.model_name = model_name

        self.is_trained = saved_data  # if new splits are generated, model needs retraining
        self.get_model()  # get preexisting model/create new one
        self.get_splits(saved_data, subset, test_split, val_split, batch_size)

    def get_model(self):
        """
        Creates and compiles a SSD300 model.
        :return: SSD300 model
        """

        optimizer = SGD(learning_rate=0.001,
                        momentum=0.5)
        loss = MultiBoxLoss()
        metrics = {'boxes': [loss.localization,
                             loss.positive_classification,
                             loss.negative_classification]}

        if exists(f"models/{self.model_name}"):  # load previous model
            self.model = load_model("models/model",
                                    custom_objects={'sgd': optimizer,
                                                    'compute_loss': loss.compute_loss,
                                                    'localization': loss.localization,
                                                    'positive_classification': loss.positive_classification,
                                                    'negative_classification': loss.negative_classification})
            self.model.prior_boxes = pickle.load(open("models/prior_boxes.p", "rb"))
        else:  # create new model
            self.model = SSD300(num_classes=len(class_names), base_weights='VGG', head_weights=None)
            self.model.compile(optimizer=optimizer, loss=loss.compute_loss, metrics=metrics)

    def get_splits(self, use_saved, subset, test_split, val_split, batch_size):
        """
        Gets d_train/d_test splits of the data.
        :param subset: How much of each split to use.
        :param batch_size: Training/validation batch sizes
        :param test_split: Size of test split
        :param val_split: Size of validation split
        :param use_saved: If True, uses saved splits instead of making new ones.
        :return: Processors for train/test splits
        """
        self.d_train, self.d_val, self.d_test = caltech(use_saved, subset, test_split, val_split, batch_size)

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
            # generate model name if none provided or name taken
            if self.model_name is None or isdir(f"models/{self.model_name}"):
                i = 0
                while isdir(f"models/model{i}"):
                    i += 1
                self.model_name = f"model{i}"

            # fit model
            history = self.model.fit(self.d_train, callbacks=callbacks, epochs=epochs, validation_data=self.d_val)

            # save model params and mark as trained
            self.model.save("models/model")
            self.is_trained = True
            pickle.dump(self.model.prior_boxes, open(f"models/{self.model_name}/prior_boxes.p", "wb"))
            return history
        else:
            print("Model is already trained!")

            return None

    def predict_model(self, img, fp=True, threshold=0.5, nms=0.5):
        """
        Uses model to make a prediction with the given image.
        :param nms: NMS threshold
        :param threshold: IoU threshold
        :param img: Image or image filepath
        :param fp: Set to True if img is a filepath, otherwise False.
        :return: BBox prediction
        """
        image = load_image(img) if fp else img
        detector = DetectSingleShot(self.model, class_names, threshold, nms, draw=True)
        results = detector(image)
        return results

    def evaluate(self, threshold=0.5, nms=0.5):
        """
        :param threshold: score threshold
        :param nms: NMS threshold
        Evaluates model using test dataset.
        :return: Dict with results
        """

        detector = DetectSingleShot(self.model, class_names, threshold, nms, draw=True)
        results = evaluateMAP(detector, self.d_test, class_labels, iou_thresh=.3)
        return results

    def show_results(self, k=None):
        """
        Makes predictions on random samples from the test dataset and displays them.
        :param k: Number of predictions to make
        """
        if not k or k > len(self.d_test):
            k = len(self.d_test)
        draw_boxes = ShowImage()

        # visualize all images that have bounding boxes
        for i, d in enumerate(sample(self.d_test, k=k)):
            if not i % int(k / 20):
                print(f"{i}/{k}")
            fp = d["image"]

            results = self.predict_model(fp)
            if not results["boxes2D"]:
                continue

            draw_boxes(results["image"])


def pipeline(saved_data=True, model_name="model", subset=1, epochs=10):
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

    print(trainer.evaluate())

    if hist:
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.legend()
        plt.show()

    # draw results
    trainer.show_results(k=100)


if __name__ == "__main__":

    pipeline(saved_data=True, model_name="model", subset=1, epochs=10)

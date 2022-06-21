from random import sample
from keras.optimizers import SGD
from paz.backend.image import load_image, GREEN
from paz.optimization import MultiBoxLoss
from paz.processors import ShowImage, DenormalizeBoxes2D, DrawBoxes2D, ToBoxes2D

import pickle
from keras.models import load_model
from os.path import exists, isdir

from paz.evaluation import evaluateMAP
from paz.pipelines.detection import DetectSingleShot
from generate_caltech_dict import class_labels, class_names
from dropout_detect import DetectSingleShotDropout


class Trainer:
    """
    Trains model to data.
    """

    def __init__(self, splits, model=None):
        """
        Initializes model and train/test splits.
        :param model: Model (keras model) or model name (str).
        """
        self.d_train, self.d_val, self.d_test = splits

        self.model = None
        self.model_name = None

        if self.model_name is not None:  # get preexisting model/create new one
            self.init_model(model)

    def init_model(self, model=None, model_name=None):
        """
        Creates and compiles a SSD300 model.
        :param model: Model name (located in models dir) or a model to be used with trainer.
        :param model_name: Optional name to give model.
        :return: SSD300 model
        """

        optimizer = SGD(learning_rate=0.001,
                        momentum=0.5)

        loss = MultiBoxLoss()
        metrics = {'boxes': [loss.localization,
                             loss.positive_classification,
                             loss.negative_classification]}

        if type(model) is str and exists(f"models/{model}"):  # load previous model
            print("=== LOADING PREVIOUS MODEL")
            self.model = load_model(f"models/{model}",
                                    custom_objects={'sgd': optimizer,
                                                    'compute_loss': loss.compute_loss,
                                                    'localization': loss.localization,
                                                    'positive_classification': loss.positive_classification,
                                                    'negative_classification': loss.negative_classification})
            self.model.prior_boxes = pickle.load(open(f"models/{self.model_name}/prior_boxes.p", "rb"))
            self.model_name = model_name if model_name else model
        elif type(model) is not str and model is not None:  # use provided model
            print("=== USING PROVIDED MODEL")
            self.model = model
            self.model_name = model_name
            self.model.compile(optimizer=optimizer, loss=loss.compute_loss, metrics=metrics)
        else:
            if type(model) is str:
                raise FileNotFoundError("model must be a valid path!")
            else:
                raise TypeError("model argument must be a valid model name or a valid keras model!")

    def train(self, callbacks=None, epochs=10):
        """
        Trains model on given data or retrieves trained model.
        callbacks: List of callbacks to use
        epochs: Number of epochs to train for
        """

        # check for callbacks
        if callbacks is None:
            callbacks = []

        # give model a name if none is set yet
        if self.model_name is None or isdir(f"models/{self.model_name}"):
            i = 0
            while isdir(f"models/{self.model_name}{i}"):
                i += 1
            self.model_name = f"{self.model_name}{i}"

        # fit model
        history = self.model.fit(self.d_train, callbacks=callbacks, epochs=epochs, validation_data=self.d_val)

        # save model params and mark as trained
        self.model.save(f"models/{self.model_name}")
        pickle.dump(self.model.prior_boxes, open(f"models/{self.model_name}/prior_boxes.p", "wb"))

        return history

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
        detector = DetectSingleShot(self.model, class_names, threshold, nms, draw=False)
        results = detector(image)
        return results

    def evaluate(self, threshold=0.5, nms=0.5):
        """
        :param threshold: score threshold
        :param nms: NMS threshold
        Evaluates model using test dataset.
        :return: Dict with results
        """

        names = class_names
        labels = class_labels
        detector = DetectSingleShot(self.model, names, threshold, nms, draw=False)
        results = evaluateMAP(detector, self.d_test, labels, iou_thresh=.3)
        return results

    def show_results(self, k=None, show_truths=False, score_thresh=.5, nms=.5):
        """
        Makes predictions on random samples from the test dataset and displays them.
        :param nms:
        :param score_thresh:
        :param k: Number of predictions to make
        :param show_truths: If True, displays the correct annotations alongside the predictions.
        """
        if not k or k > len(self.d_test):
            k = len(self.d_test)

        # visualize all images that have bounding boxes
        for i, d in enumerate(sample(self.d_test, k=k)):
            if not i % int(k / 20):
                print(f"{i}/{k}")
            fp = d["image"]

            results = self.predict_model(fp, threshold=score_thresh, nms=nms)

            show_image = ShowImage()
            draw_pred = DrawBoxes2D(class_names)

            image = results["image"]
            draw_img = draw_pred(image, [])

            # draw boxes
            draw_img = draw_pred(draw_img, results["boxes2D"])

            # draw truths if requested
            if show_truths:
                to_boxes2D = ToBoxes2D(class_names)
                denormalize = DenormalizeBoxes2D()
                boxes2D = to_boxes2D(d["boxes"])
                boxes2D = denormalize(image, boxes2D)
                draw_truths = DrawBoxes2D(class_names, colors=[list(GREEN), list(GREEN)])
                draw_img = draw_truths(draw_img, boxes2D)
            show_image(draw_img)


class DropoutTrainer(Trainer):

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
        detector = DetectSingleShotDropout(self.model, class_names, threshold, nms, draw=False)
        results = detector(image)
        return results

    def show_results(self, k=None, show_truths=False, score_thresh=.5, nms=.5):
        """
        Makes predictions on random samples from the test dataset and displays them.
        :param nms:
        :param score_thresh:
        :param k: Number of predictions to make
        :param show_truths: If True, displays the correct annotations alongside the predictions.
        """
        # limit number of predictions to be between 0 and the length of the test set
        if not k or k > len(self.d_test):
            k = len(self.d_test)

        # visualize all images
        for i, d in enumerate(sample(self.d_test, k=k)):
            if not i % int(k / 20):
                print(f"{i}/{k}")
            fp = d["image"]

            results = self.predict_model(fp, threshold=score_thresh, nms=nms)
            show_image = ShowImage()
            draw_mean = DrawBoxes2D(class_names)

            image = results["image"]
            draw_img = draw_mean(image, [])

            # draw boxes
            draw_img = draw_mean(draw_img, results["boxes2D"])

            # draw truths if requested
            if show_truths:
                to_boxes2D = ToBoxes2D(class_names)
                denormalize = DenormalizeBoxes2D()
                boxes2D = to_boxes2D(d["boxes"])
                boxes2D = denormalize(image, boxes2D)
                draw_truths = DrawBoxes2D(class_names, colors=[list(GREEN), list(GREEN)])
                draw_img = draw_truths(draw_img, boxes2D)
            show_image(draw_img)

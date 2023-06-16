import json
from os import mkdir

from cv2 import imwrite
from keras.optimizers import SGD
from paz.backend.image import load_image, convert_color_space, RGB2BGR
from paz.models.detection.utils import create_prior_boxes
from paz.optimization import MultiBoxLoss
from paz.processors import ShowImage, DrawBoxes2D

from keras.models import load_model
from os.path import exists

from paz.evaluation import evaluateMAP
from paz.pipelines.detection import DetectSingleShot
from dataset_processing.generate_caltech_dict import class_labels, class_names
from backend.dropout.dropout_pipeline import StochasticDetectSingleShot
from backend.dropout.ssd_dropout import SSD300_dropout
from backend.dropout.dropout_draw import DrawBoxesDropout


def default_ssd_parameters():
    """Initializes SSD parameters.

    Returns: SGD optimizer, MultiBoxLoss, and loss metrics.
    """
    optimizer = SGD(learning_rate=0.001,
                    momentum=0.6)

    loss = MultiBoxLoss(alpha=1.0)
    metrics = {'boxes': [loss.localization,
                         loss.positive_classification,
                         loss.negative_classification]}
    return optimizer, loss, metrics


def init_ssd(rate=0.3):
    """Initializes SSD300 model using default parameters.

    Args:
        rate: Model dropout rate.

    Returns:
        Untrained SSD300 model.
    """
    ssd = SSD300_dropout(num_classes=len(class_names), base_weights='VGG', head_weights=None, prob=rate)

    optimizer, loss, metrics = default_ssd_parameters()

    ssd.compile(optimizer=optimizer, loss=loss.compute_loss, metrics=metrics)

    return ssd


def make_deterministic(model):
    """Initializes an SSD300-Dropout model with dropout rate 0 and copies weights of given model to it.

    Args:
        model: SSD300-Dropout model.

    Returns:
        Deterministic SSD300-Dropout model
    """
    det_model = init_ssd(rate=0)
    det_model.set_weights(model.get_weights())
    return det_model


def load_from_path(model_path):
    """Loads SSD300 model with default model parameters.

    Args:
        model_path: Path to model directory.

    Returns:
        SSD300 model
    """

    optimizer, loss, metrics = default_ssd_parameters()

    model = load_model(model_path,
                       custom_objects={'sgd': optimizer,
                                       'compute_loss': loss.compute_loss,
                                       'localization': loss.localization,
                                       'positive_classification': loss.positive_classification,
                                       'negative_classification': loss.negative_classification})
    model.prior_boxes = create_prior_boxes(configuration_name='VOC')
    return model


def train_model(model, d_train, d_val, save_dir, callbacks=None, epochs=10):
    """
    Trains model on given data and saves it.

    Args:
        model: Model to be trained.
        callbacks: List of callbacks to use
        epochs: Number of epochs to train for
        d_train: Training dataset processor
        d_val: Validation dataset processor
        save_dir: Directory to save model in.

    Returns:
        History object containing model training history
    """

    # check for callbacks
    if callbacks is None:
        callbacks = []

    # fit model
    history = model.fit(d_train, callbacks=callbacks, epochs=epochs, validation_data=d_val)

    # save model params and mark as trained
    model.save(save_dir)

    with open(f'{save_dir}/train.json', 'w') as fp:
        json.dump(history.history, fp)

    return history


def predict_ssd(model, img_path, threshold=0.5, nms=0.5, samples=0, std_thresh=0):
    """
    Uses model to make a prediction given image filepaths.

    Args:
        model: SSD model
        nms: NMS threshold
        threshold: IoU threshold
        img_path: Image filepaths. List.
        samples: Number of samples to take from model. Used for stochastic models. Int.
        std_thresh: For stochastic model. Min IoU of mean box/std box per prediction. Used to filter boxes based on uncertainty.

    Returns:
        Dictionary of bbox predictions per image.
    """

    names = class_names

    results = []

    for imp in img_path:
        img = load_image(imp)

        if samples > 1:  # stochastic case
            detector = StochasticDetectSingleShot(model, names, threshold, nms, draw=False, samples=samples,
                                                  std_thresh=std_thresh)
        else:  # deterministic case
            detector = DetectSingleShot(model, names, threshold, nms, draw=False)

        res = detector(img)

        results.append(res)

    return results


def evaluate(model, dataset, score_thresh=0.45, nms=0.5, iou=0.5, samples=0, std_thresh=0):
    """
    Evaluates PAZ SSD model using test dataset.

    Args:
        model: SSD model.
        dataset: Evaluation dataset. Dict with keys {'image', 'boxes'}
        score_thresh: score threshold. Float between 0,1.
        nms: NMS threshold. Float between 0,1.
        iou: IoU threshold for MAP evaluation. Float between 0,1.
        samples: Number of samples to take from model. Used for stochastic models. Int.
        std_thresh: For stochastic model. Min IoU of mean box/std box per prediction. Used to filter boxes based on
        uncertainty.

    Returns:
        Dict with keys ['image','boxes2D'] for deterministic model, ['image', 'boxes2D','std'] for stochastic.
    """

    names = class_names
    labels = class_labels

    if samples > 1:  # stochastic case
        detector = StochasticDetectSingleShot(model, names, score_thresh, nms, draw=False, samples=samples,
                                              std_thresh=std_thresh)
    else:  # deterministic case
        detector = DetectSingleShot(model, names, score_thresh, nms, draw=False)

    meanAP = evaluateMAP(detector, dataset, labels, iou_thresh=iou)

    return meanAP


def save_image(img, img_name, save_dir):
    """ Saves image in given directory with given name.

    Args:
        img: OpenCV image.
        img_name: Image save name.
        save_dir: Image save directory.
    """
    if not exists(save_dir):
        mkdir(save_dir)

    img = convert_color_space(img, RGB2BGR)
    imwrite(f"{save_dir}/{img_name}", img)


def draw_predictions(draw_set, show_results=True, save_dir=False, display_std=False):
    """Draw a set of predictions.

    Args:
        draw_set: Set of predictions to draw.
        save_dir: Filepath at which to save results. If None, results are not saved.
        show_results: If True
        display_std: If True, visualizes model uncertainty. predictions must be stochastic.
    """

    # visualize all images that have bounding boxes

    for i, d in enumerate(draw_set):

        show_image = ShowImage()
        draw_pred = DrawBoxes2D(class_names, scale=0.5)

        image = d["image"]

        draw_img = draw_pred(image, [])

        # draw
        draw_img = draw_pred(draw_img, d["boxes2D"])

        # if True, displays std-adjusted box
        if display_std:
            draw_stds = DrawBoxesDropout(class_names, colors=draw_pred.colors, scale=0)
            draw_img = draw_stds(draw_img, d["std"])

        # if True, shows image
        if show_results:
            show_image(draw_img)

        # saves image in directory if provided
        if save_dir:
            img_name = f"img{i}.jpg"
            save_image(draw_img, img_name, save_dir)
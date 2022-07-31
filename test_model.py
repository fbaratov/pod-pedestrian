import numpy as np

from backend.dataset_processing.prep_dataset import retrieve_splits
from backend.model_utils import *


def evaluate_model_map(model_type, model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=0, std_thresh=0.):
    """
    Function for evaluating model mAP on test set.

    Args:
        model_type: Model type. "stochastic" or "deterministic".
        model: PAZ SSD model or variant.
        test_set: Set on which to evaluate mAP. Dict with keys "images" and "boxes".
        score_thresh: Prediction confidence threshold. Float, [0,1].
        nms: NMS IoU threshold. Float, [0,1].
        iou: Match IoU threshold. Float, [0,1].
        samples: Number of samples to make for stochastic model. Int.
        std_thresh: Mean box and STD-adjusted box IoU threshold. Float, [0,1].

    Returns:
        Dictionary of mAP evaluation results
    """
    if model_type == 'deterministic':
        model = make_deterministic(model)
        results = evaluate(model, test_set, score_thresh=score_thresh, nms=nms, iou=iou, samples=0)
    elif model_type == 'stochastic':
        results = evaluate(model, test_set, score_thresh=score_thresh, nms=nms, iou=iou, samples=samples,
                           std_thresh=std_thresh)
    else:
        raise ValueError("Incorrect model type!")

    return results


if __name__ == "__main__":
    _, _, test_set = retrieve_splits("pickle/all_classes_70_15_15")
    for i, d in enumerate(test_set):
        for j, b in enumerate(d['boxes']):
            d['boxes'][j] = np.array([b[0] * 640, b[1] * 480, b[2] * 640, b[3] * 480, b[4]])

    model = load_from_path("models/model_1")

    deterministic = evaluate_model_map("deterministic", model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=5, std_thresh=0)
    stochastic = evaluate_model_map("stochastic", model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=5, std_thresh=0)
    sto_filtered = evaluate_model_map("stochastic", model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=5, std_thresh=0.85)

    print(deterministic)
    print(stochastic)
    print(sto_filtered)

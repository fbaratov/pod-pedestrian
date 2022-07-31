from random import random, sample

from prep_dataset import retrieve_splits
from trainer import *


def visualize_predictions(model_type, model, pred_set, save_dir=None, show_results=True, score_thresh=.5, nms=.45,
                          samples=0, std_thresh=0.):
    if model_type == "deterministic":
        model = make_deterministic(model)

    if model_type == 'deterministic':
        draw_set = predict_ssd(model, pred_set, threshold=0.5, nms=0.5, samples=0)
    elif model_type == 'stochastic':
        draw_set = predict_ssd(model, pred_set, threshold=score_thresh, nms=nms, samples=samples, std_thresh=std_thresh)
    else:
        raise ValueError("Incorrect model type!")

    draw_predictions(draw_set, show_results=show_results, save_dir=save_dir, display_std=(model_type == "stochastic"))


if __name__ == "__main__":
    _, _, test_set = retrieve_splits("pickle/all_classes_70_15_15")
    model = load_from_path("models/model_1")

    pred_set = sample([d["image"] for d in test_set], k=1000)

    det_dir = "predictions/deterministic"

    visualize_predictions("deterministic",
                          model,
                          pred_set,
                          save_dir=det_dir,
                          show_results=False,
                          samples=5)

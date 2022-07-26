from random import sample

import numpy as np

from prep_dataset import retrieve_splits
from trainer import *
import argparse


def evaluate_model_map(model_type, model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=0, std_thresh=0.):

    if model_type == 'deterministic':
        model = make_deterministic(model)
        results = evaluate(model, test_set, score_thresh=score_thresh, nms=nms, iou=iou, samples=0)
    elif model_type == 'stochastic':
        results = evaluate(model, test_set, score_thresh=score_thresh, nms=nms, iou=iou, samples=samples,
                           std_thresh=std_thresh)
    else:
        raise ValueError("Incorrect model type!")

    print(results)


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='model type, either \'deterministic\', \'stochastic\'')
    parser.add_argument('model_path', type=str, help='model directory path')
    parser.add_argument('--split_path', type=str, help='test split directory path')
    parser.add_argument('--score_threshold', type=float, help='score threshold', default=0.3)
    parser.add_argument('--nms', type=float, help='non-maximum suppression threshold', default=0.6)
    parser.add_argument('--map', type=bool, help='if True, evaluates model mean average precision (mAP)', default=False)
    parser.add_argument('--k', type=int, help='number of results to show', default=100)
    parser.add_argument('--show_truths', type=bool, help='if True, draws ground truths on results', default=False)
    parser.add_argument('--show_results', type=bool, help='if True, displays examples of predictions', default=False)
    parser.add_argument('--save_results', type=bool, help='if True, saves images to predictions folder', default=False)
    parser.add_argument('--compare', type=bool,
                        help='if True, compares two provided models by evaluating them on the same set', default=False)
    parser.add_argument('--second_model_type', type=str,
                        help='second model type for comparison, either \'baseline\', \'dropout\'', default=None)
    parser.add_argument('--second_model_name', type=str, help='second model directory name for comparison',
                        default=None)

    a = parser.parse_args()"""

    _, _, test_set = retrieve_splits("pickle/all_classes_70_15_15")
    for i, d in enumerate(test_set):
        for j, b in enumerate(d['boxes']):
            d['boxes'][j] = np.array([b[0] * 640, b[1] * 480, b[2] * 640, b[3] * 480, b[4]])

    model = load_from_path("models/dropout_model_full_0")

    evaluate_model_map("stochastic", model, test_set, score_thresh=.5, nms=.45, iou=0.5, samples=5, std_thresh=0.75)

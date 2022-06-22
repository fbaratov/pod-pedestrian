import numpy as np
from paz.abstract import Processor
from paz.backend.boxes import apply_non_max_suppression
from paz.processors import NonMaximumSuppressionPerClass


def nms_per_class_dropout(mean_data, std_data, nms_thresh=.45, conf_thresh=0.01, top_k=200):
    """Applies non-maximum-suppression per class.
    # Arguments
        box_data: Numpy array of shape `(num_prior_boxes, 4 + num_classes)`.
        nsm_thresh: Float. Non-maximum suppression threshold.
        conf_thresh: Float. Filter scores with a lower confidence value before
            performing non-maximum supression.
        top_k: Integer. Maximum number of boxes per class outputted by nms.

    Returns
        Numpy array of shape `(num_classes, top_k, 5)` and list of selected indices.
    """
    decoded_boxes, class_predictions = mean_data[:, :4], mean_data[:, 4:]
    decoded_stds = std_data[:, :4]

    num_classes = class_predictions.shape[1]
    out_mean = np.zeros((num_classes, top_k, 5))
    out_std = np.zeros((num_classes, top_k, 5))

    selection = []
    # skip the background class (start counter in 1)
    for class_arg in range(1, num_classes):
        # confidence mask
        conf_mask = class_predictions[:, class_arg] >= conf_thresh
        scores = class_predictions[:, class_arg][conf_mask]

        # skip if no boxes found
        if len(scores) == 0:
            continue

        # select boxes
        means = decoded_boxes[conf_mask]
        indices, count = apply_non_max_suppression(
            means, scores, nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)
        selected_indices = indices[:count]

        selection += list(selected_indices)  # note selected indices

        selections = np.concatenate(
            (means[selected_indices], scores[selected_indices]), axis=1)
        out_mean[class_arg, :count, :] = selections

        stds = decoded_stds
        selections = np.concatenate(
            (stds[selected_indices], scores[selected_indices]), axis=1)
        out_mean[class_arg, :count, :] = selections

    return out_mean, out_std


class NMSPerClassDropout(NonMaximumSuppressionPerClass):
    """Applies non maximum suppression for dropout predictions and indices in vector.

    # Arguments
        nms_thresh: Float between [0, 1].
        conf_thresh: Float between [0, 1].
    """

    def call(self, mean_data, std_data):
        means, stds = nms_per_class_dropout(mean_data, std_data, self.nms_thresh, self.conf_thresh)

        print(mean_data.shape, std_data.shape)
        return means, stds


class CreateSTDBoxes(Processor):

    def call(self, means, stds):
        std_boxes = []

        # filter list to only compatible boxes
        print(len(means), len(stds))

        # create std boxes
        for m, std in zip(means, stds):
            x0 = m[0] - std[0]
            y0 = m[1] - std[1]
            x1 = m[2] + std[2]
            y1 = m[3] + std[2]
            coords = np.array([x0, y0, x1, y1])
            std_boxes.append(coords)

        # convert to np.array
        std_boxes = np.array(std_boxes)

        return std_boxes

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
    decoded_means, class_predictions = mean_data[:, :4], mean_data[:, 4:]
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

        # apply confidence mask
        means = decoded_means[conf_mask]
        stds = decoded_stds[conf_mask]

        # apply nms based on means
        indices, count = apply_non_max_suppression(
            means, scores, nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)

        selected_indices = indices[:count]

        # selection += list(selected_indices)  # note selected indices

        # save selected means and stds
        selected_means = np.concatenate(
            (means[selected_indices], scores[selected_indices]), axis=1)
        out_mean[class_arg, :count, :] = selected_means

        selected_stds = np.concatenate(
            (stds[selected_indices], scores[selected_indices]), axis=1)
        out_std[class_arg, :count, :] = selected_stds

    return out_mean, out_std


class NMSPerClassDropout(NonMaximumSuppressionPerClass):
    """Applies non maximum suppression for dropout predictions and indices in vector.

    # Arguments
        nms_thresh: Float between [0, 1].
        conf_thresh: Float between [0, 1].
    """

    def call(self, mean_data, std_data):
        means, stds = nms_per_class_dropout(mean_data, std_data, self.nms_thresh, self.conf_thresh)
        return means, stds


class CreateSTDBoxes(Processor):

    def call(self, means, stds):
        std_boxes = []

        # filter list to only compatible boxes
        pairs = list(zip(means, stds))

        # create std boxes
        for i, p in enumerate(pairs):
            m, std = p
            x0 = m[0] - std[0]
            y0 = m[1] - std[1]
            x1 = m[2] + std[2]
            y1 = m[3] + std[3]
            coords = np.array([x0, y0, x1, y1, std[4], std[5]])
            std_boxes.append(coords)

        return np.array(std_boxes)

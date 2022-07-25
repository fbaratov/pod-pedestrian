import numpy as np
from paz.abstract import Processor
from paz.backend.boxes import apply_non_max_suppression
from paz.processors import NonMaximumSuppressionPerClass, Predict


def nms_per_class_sampling(mean_data, std_data, nms_thresh=.45, conf_thresh=0.01, top_k=200):
    """Applies non-maximum-suppression per class.
    # Arguments
        box_data: Numpy array of shape `(num_prior_boxes, 4 + num_classes)`.
        nsm_thresh: Float. Non-maximum suppression threshold.
        conf_thresh: Float. Filter scores with a lower confidence value before
            performing non-maximum supression.
        top_k: Integer. Maximum number of boxes per class outputted by nms.

    Returns
        Two numpy arrays of shape `(num_classes, top_k, 5)`.
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

        # save selected means and stds
        selected_means = np.concatenate(
            (means[selected_indices], scores[selected_indices]), axis=1)
        out_mean[class_arg, :count, :] = selected_means

        selected_stds = np.concatenate(
            (stds[selected_indices], scores[selected_indices]), axis=1)
        out_std[class_arg, :count, :] = selected_stds

    return out_mean, out_std


class NMSPerClassSampling(NonMaximumSuppressionPerClass):
    """Applies non maximum suppression for dropout predictions and indices in vector.

    # Arguments
        nms_thresh: Float between [0, 1].
        conf_thresh: Float between [0, 1].
    """

    def call(self, mean_data, std_data):
        means, stds = nms_per_class_sampling(mean_data, std_data, self.nms_thresh, self.conf_thresh)
        return means, stds


class STDFilter(Processor):
    def __init__(self, name=None, iou=0.85):
        super(STDFilter, self).__init__()
        self.name = name
        self.iou = iou
    
    def call(self, means, stds):
        filtered_means, filtered_stds = [], []
        for m, s in zip(means, stds):
            mc = m.coordinates
            mx = mc[2] - mc[0]
            my = mc[3] - mc[1]
            ma = mx * my

            sc = s.coordinates
            sx = sc[2] - sc[0]
            sy = sc[3] - sc[1]
            sa = sx * sy

            if ma / sa > self.iou:
                filtered_means.append(m)
                filtered_stds.append(s)

        return filtered_means, filtered_stds


class CreateSTDBoxes(Processor):
    def call(self, means, stds):
        std_boxes = []

        # filter list to only compatible boxes
        pairs = list(zip(means, stds))

        # create std boxes
        for i, p in enumerate(pairs):
            m, std = p

            x = m[0]
            y = m[1]
            W = m[2] + std[0] + std[2]
            H = m[3] + std[1] + std[3]
            coords = np.array([x, y, W, H])

            bbox = np.concatenate([coords, std[4:]], axis=0)
            std_boxes.append(bbox)

        return np.array(std_boxes)


class PredictBoxesSampling(Predict):

    def call(self, x, k=100):
        # preprocess x
        if self.preprocess is not None:
            x = self.preprocess(x)

        # make predictions and concatenate
        regr = None
        classify = None

        for _ in range(k):
            # get regression and classification from prediction
            y = self.model.predict(x)
            y_regr = y[:, :, :4]
            y_classify = y[:, :, 4:]

            if regr is None:  # initialize regr/classification arrays
                regr = np.empty(y_regr[1:, :, :].shape)
                classify = np.empty(y_classify[1:, :, :].shape)

            # concatenate regression/classification predictions to arrays
            regr = np.concatenate([regr, y_regr], axis=0)
            classify = np.concatenate([classify, y_classify], axis=0)

        # calculate distribution values
        regr_mean = np.mean(regr, axis=0)
        regr_std = np.std(regr, axis=0)
        classify_mean = np.mean(classify, axis=0)

        # concatenate
        means = np.concatenate([regr_mean, classify_mean], axis=1)
        stds = np.concatenate([regr_std, classify_mean], axis=1)

        # postprocess
        bbox_norm = self.postprocess(means, stds)

        return bbox_norm

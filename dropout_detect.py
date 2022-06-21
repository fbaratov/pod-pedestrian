import numpy as np
from paz.abstract import SequentialProcessor, Processor
import paz.processors as pr
from paz.pipelines import DetectSingleShot

from generate_caltech_dict import class_names, class_labels


class DetectSingleShotDropout(DetectSingleShot):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.
    """

    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN, variances=[0.1, 0.1, 0.2, 0.2],
                 draw=True):

        super(DetectSingleShotDropout, self).__init__(model, class_names, score_thresh, nms_thresh,
                                                      mean, variances, draw)

        self.postprocessing = self.predict.postprocess
        self.predict.postprocess = None
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'std'])

    def call(self, image, k=100):
        # make predictions and concatenate
        regr = np.empty((0, 8732, 4))
        classify = np.empty((0, 8732, len(class_names)))

        for _ in range(k):
            # get regression and classification from prediction
            pred = self.predict(image)
            pred_regr = pred[0]
            pred_classify = pred[1]

            # concatenate regression/classification predictions to arrays
            regr = np.concatenate([regr, pred_regr], axis=0)
            classify = np.concatenate([classify, pred_classify], axis=0)

        # calculate regression mean
        regr_mean = np.mean(regr, axis=0)
        print(regr.shape, "=>", regr_mean.shape)

        # calculate regression variance
        regr_std = np.std(regr, axis=0)
        print(regr.shape, "=>", regr_std.shape)

        # get classification mean
        classify_mean = np.mean(classify, axis=0)
        print(classify.shape, "=>", classify_mean.shape)

        # concatenate and postprocess
        bbox_mean = np.concatenate([regr_mean, classify_mean], axis=1)
        print(bbox_mean.shape)
        boxes2D_normalized = self.postprocessing(bbox_mean)

        # denormalize boxes and add to list
        boxes2D = []
        for box in boxes2D_normalized:
            try:
                denorm_box = self.denormalize(image, [box])

                # consider only boxes with valid coordinates
                for db in denorm_box:
                    x0, y0, x1, y1 = db.coordinates
                    if (0 <= x0 <= image.shape[0] and 0 <= y0 <= image.shape[1] and
                            0 <= x1 <= image.shape[0] and 0 <= y1 <= image.shape[1]):
                        boxes2D.append(db)
            except ValueError as e:
                boxes2D_normalized.remove(box)
                print("Failure to normalize mean box:", box)

        if self.draw: # draw boxes if requested
            image = self.draw_boxes2D(image, boxes2D)

        return self.wrap(image, boxes2D, regr_std)

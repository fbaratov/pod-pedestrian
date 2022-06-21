import numpy as np
from paz.abstract import SequentialProcessor, Processor
import paz.processors as pr


class DetectSingleShotDropout(Processor):
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
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw

        super(DetectSingleShotDropout, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(self.model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.SubtractMeanImage(mean),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])

        self.postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(self.model.prior_boxes, self.variances),
             pr.NonMaximumSuppressionPerClass(self.nms_thresh),
             pr.FilterBoxes(self.class_names, self.score_thresh)])

        self.predict = pr.Predict(self.model, preprocessing, postprocess=None)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'std'])

    def call(self, image, k=100):
        regr = []
        classify = []
        for _ in range(k):
            pred = self.predict(image)
            regr.append(pred[0])
            classify.append(pred[1])

        # calculate mean
        regr_mean = np.mean(regr, axis=0)

        # calculate variance
        regr_var = 0
        for i in regr:
            regr_var += np.add(np.var(i, axis=0), np.power(np.mean(i), 2))
        regr_var /= len(regr)
        regr_var -= np.power(regr_mean, 2)

        classify_mean = np.mean(classify, axis=0)

        # concatenate and postprocess

        # denormalize

        boxes2D = []
        for fucker in [regr_mean, regr_var]:
            bbox_mean = np.concatenate([fucker, classify_mean], axis=2)
            boxes2D_normalized = self.postprocessing(bbox_mean)

            # add boxes to box list
            for box in boxes2D_normalized:
                try:
                    denorm_box = self.denormalize(image, [box])
                    # filter all
                    for db in denorm_box:
                        x0, y0, x1, y1 = db.coordinates
                        if (0 <= x0 <= image.shape[0] and
                                0 <= y0 <= image.shape[1] and
                                0 <= x1 <= image.shape[0] and
                                0 <= y1 <= image.shape[1]):  # remove all negative coordinates
                            boxes2D.append(db)
                            print(db, end=" ")
                except ValueError as e:
                    boxes2D_normalized.remove(box)
                    print("Failure to normalize mean box:", box)
            print("")

        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D, regr_var)

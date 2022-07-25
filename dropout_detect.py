from copy import deepcopy

import numpy as np
from paz.abstract import SequentialProcessor
import paz.processors as pr
from paz.pipelines import DetectSingleShot

from dropout_processors import *


class StochasticDetectSingleShot(DetectSingleShot):
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
                 draw=True, samples=5, std_thresh=0):

        super(StochasticDetectSingleShot, self).__init__(model, class_names, score_thresh, nms_thresh,
                                                         mean, variances, draw)
        self.samples = samples
        self.std_thresh = std_thresh

        # construct postprocessor
        decode = SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(self.model.prior_boxes, self.variances)])

        filter_boxes = pr.FilterBoxes(self.class_names, self.score_thresh)

        postprocessing = SequentialProcessor([  # input: means before postproc, stds before postproc
            # construct std boxes
            pr.ControlMap(CreateSTDBoxes(), intro_indices=[0, 1], outro_indices=[1], keep={0: 0}),
            # decode boxes
            pr.ControlMap(decode, intro_indices=[0], outro_indices=[0]),
            pr.ControlMap(decode, intro_indices=[1], outro_indices=[1]),
            # apply NMS based on means
            pr.ControlMap(NMSPerClassSampling(self.nms_thresh), intro_indices=[0, 1], outro_indices=[0, 1]),
            # filter boxes
            pr.ControlMap(filter_boxes, intro_indices=[0], outro_indices=[0]),
            pr.ControlMap(filter_boxes, intro_indices=[1], outro_indices=[1]),
            pr.ControlMap(STDFilter(iou=std_thresh), intro_indices=[0, 1], outro_indices=[0, 1])
        ])

        # create predictor and output wrap
        self.predict = PredictBoxesSampling(self.model, self.predict.preprocess, postprocessing)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'std'])

    def call(self, image):
        # make predictions
        bbox_means_norm, bbox_stds_norm = self.predict(image, self.samples)

        # denormalize boxes and add to list
        box_means = []
        box_stds = []
        for m, std in zip(bbox_means_norm, bbox_stds_norm):
            try:
                mean_denorm, std_denorm = self.denormalize(image, [m, std])
                box_means.append(mean_denorm)
                box_stds.append(std_denorm)
            except ValueError as e:
                print("Failure to normalize mean-std combo:", m, std)

        if self.draw:  # draw boxes if requested
            image = self.draw_boxes2D(image, box_means)
            image = self.draw_boxes2D(image, box_stds)

        return self.wrap(image, box_means, box_stds)

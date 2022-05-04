"""
This file contains a modified version of the AugmentDetection class called AugmentCaltech,
mainly used to debug the processor by seeing where it runs into issues.
Encoding currently causes the xmax/ymax coordinates to be smaller than their respective minimums, which isn't meant
to happen. Therefore it has been commented out until I can figure out why it is an issue.
"""

import numpy as np
from itertools import product
from paz.abstract import SequentialProcessor
import paz.processors as pr
from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentImage, PreprocessImage, AugmentBoxes


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """

    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU), )
        # self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentCaltech(SequentialProcessor):
    """Augment boxes and images for object detection.

    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """

    def __init__(self, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentCaltech, self).__init__()
        # image processors
        # self.augment_image = AugmentImage()
        # self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        # self.augment_boxes = AugmentBoxes()
        prior_boxes = create_prior_boxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        #if split == pr.TRAIN:
            # self.add(pr.ControlMap(self.augment_image, [0], [0]))
            # self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))

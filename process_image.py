from paz.abstract import SequentialProcessor
import paz.processors as pr


class AugmentImage(SequentialProcessor):
    """
    Adds various filters to the image as specified the constructor function.
    """
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation())
        self.add(pr.RandomHue())


class PreprocessImage(SequentialProcessor):
    """
    Preprocesses image into the size/shape required for SSD.
    """
    def __init__(self, shape=(300, 300), mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape)) # resize image to input size
        self.add(pr.CastImage(float)) # cast image to a float
        if mean is None: # normalize the image if necessary
            self.add(pr.NormalizeImage())
        else:
            self.add(pr.SubtractMeanImage(mean))

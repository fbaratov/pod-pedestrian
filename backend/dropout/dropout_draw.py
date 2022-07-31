from paz.abstract import Processor
from paz.backend.image import lincolor, put_text, draw_rectangle
from paz.processors import DrawBoxes2D


class DrawBoxesDropout(DrawBoxes2D):
    """Draws bounding boxes from Boxes2D messages. Modified for models that output mean boxes and
    uncertainty-adjusted boxes. Based on PAZ DrawBoxes2D processor.
    """

    def call(self, image, boxes2D):
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            class_name = box2D.class_name
            color = self.class_to_color[class_name]
            if self.weighted:
                color = [int(channel * box2D.score) for channel in color]
            if self.with_score:
                text = '{:0.2f}, {}'.format(box2D.score, class_name)
            if not self.with_score:
                text = '{}'.format(class_name)
            put_text(image, text, (x_min, y_min - 10), self.scale, color, 1)
            draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
        return image
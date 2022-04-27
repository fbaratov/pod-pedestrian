import os

import numpy as np
from paz.abstract import ProcessingSequence, SequentialProcessor
from paz.models.detection.utils import create_prior_boxes
from paz.pipelines import AugmentDetection
import paz.processors as pr
import paz.backend as P


def deprocess_image(image):
    image = (image + pr.BGR_IMAGENET_MEAN).astype('uint8')
    return P.image.convert_color_space(image, pr.BGR2RGB)


def extract_box_caltech(file, width=640, height=480):
    """
    Extracts bounding boxes from the provided file.
    :param file: path to annotations file.
    :param width: image width. default is 640px (caltech pedestrian dataset size)
    :param height: image height. default is 480px (caltech pedestrian dataset size)
    :returns list of bounding boxes.
    """
    model_labels = {
        "background": 0,
        "person": 1,
        "person-fa": 1,
        "person?": 1,
        "people": 2
    }

    box_data = []
    with open(file, 'r') as f:
        line = f.readline()  # skip first line, its not a bounding box
        while line:
            line = f.readline()  # get line

            if not line:  # exit condition
                break

            split = line.split(" ")
            label = split[0]
            x0, y0, width, height = [int(val) for val in split[1:5]]
            x1 = x0 + width
            y1 = y0 + height
            label_int = model_labels[label]
            box_data.append([x0, y0, x1, y1, label_int])

    return np.array(box_data)


def prep_data():
    # useful directories
    dataset_dir = "D:/.datasets/CALTECH_PEDESTRIAN/unpacked"
    img_dir = f"{dataset_dir}/images"
    annot_dir = f"{dataset_dir}/annotations"

    # collect data here
    data = []

    # match sets to images
    for set in os.listdir(annot_dir):
        print(set)
        for vdir in os.listdir(f"{annot_dir}/{set}"):
            print(f"  {vdir}")
            for annot_name in os.listdir(f"{annot_dir}/{set}/{vdir}"):
                num = annot_name[1:-4]
                img_name = f"{set}_{vdir}_0{num}.jpg"  # get image name (will have to change once image names are fixed)
                img_path = f"{img_dir}/{set}/{img_name}"  # assemble image path
                annot_path = f"{annot_dir}/{set}/{vdir}/{annot_name}"  # assemble annotation path

                # extract boxes
                boxes = extract_box_caltech(annot_path)

                if len(boxes) == 0:
                    continue

                # append to full set
                data.append({
                    "image": img_path,
                    "boxes": boxes
                })
        break
    print(data[0:10])
    return data


def caltech():
    data = prep_data()

    class_names = ['background', 'person', 'people']
    prior_boxes = create_prior_boxes()
    augmentator = AugmentDetection(prior_boxes, num_classes=len(class_names))

    draw_boxes = SequentialProcessor([
        pr.ControlMap(pr.ToBoxes2D(class_names, True), [1], [1]),
        pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}),
        pr.DrawBoxes2D(class_names),
        pr.ShowImage()])

    print('Image and boxes augmentations with generator')
    batch_size = 1
    sequence = ProcessingSequence(augmentator, batch_size, data)
    while True:
        try:
            batch = sequence.__getitem__(0)
        except:
            continue
        batch_images, batch_boxes = batch[0]['image'], batch[1]['boxes']
        image, boxes = batch_images[0], batch_boxes[0]
        image = deprocess_image(image)
        draw_boxes(image, boxes)
        break

if __name__=="__main__":
    caltech()

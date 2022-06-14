from keras.utils import plot_model as plot_model
from paz.models import SSD300

import pickle
from ssd_dropout import SSD300_dropout
from train_net import class_labels, class_names
from train_net import Trainer


def model_fn(prob=0.3):
    ssd = SSD300_dropout(num_classes=len(class_names), base_weights='VGG', head_weights=None, prob=prob)
    #ssd = SSD300(num_classes=len(class_names), base_weights='VGG', head_weights=None)
    return ssd


if __name__ == "__main__":
    model = model_fn(0.3)

    """plot_model(
        model,
        to_file="model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
    )"""

    trainer = Trainer(model_name=None,
                      saved_data=False,
                      subset=0.05,
                      batch_size=8)

    trainer.init_model(model, "dropout_model")
    trainer.train()

    model.prior_boxes = pickle.load(open('models/prior_boxes.p', 'rb'))

    trainer.model = model

    vals = trainer.show_results(k=100, show_truths=True)

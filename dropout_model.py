from ssd_dropout import SSD300_dropout
from train_net import *


def model_fn(prob=0.3):
    ssd = SSD300_dropout(num_classes=len(class_names), base_weights='VGG', head_weights=None, prob=prob)
    # ssd = SSD300(num_classes=len(class_names), base_weights='VGG', head_weights=None)
    return ssd


if __name__ == "__main__":
    model = model_fn(0.5)

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

    trainer = DropoutTrainer(model_name=None,
                             saved_data=True,
                             subset=0.05,
                             batch_size=10)

    trainer.init_model(model, "dropout_model_biggerbatch")
    trainer.train()

    model.prior_boxes = pickle.load(open('models/prior_boxes.p', 'rb'))

    trainer.show_results(k=100, show_truths=True, num_preds=50)

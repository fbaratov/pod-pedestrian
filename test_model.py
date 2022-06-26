from keras import Model
from paz.evaluation import evaluateMAP

from prep_dataset import retrieve_splits
from trainer import DropoutTrainer


def test_dropout():
    split_name = "full_set"
    model_name = "dropout_model_full_0"

    trainer = DropoutTrainer(model=model_name,
                             splits=retrieve_splits(split_name))

    # convert model to two-headed model
    model = trainer.model

    # generate predictions
    trainer.show_results(k=100, show_truths=True, score_thresh=.3, nms=.6)



if __name__ == "__main__":
    test_dropout()

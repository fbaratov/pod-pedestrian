import pickle
from os.path import exists

from paz.models.detection.ssd300 import SSD300

from train_pipeline import caltech


if __name__ == "__main__":
    gen = caltech()
    use_saved = True
    if use_saved and exists("pickle/model.p"):
        model = pickle.load(open("pickle/model.p", "rb"))
    else:
        model = SSD300(num_classes=2, base_weights=None, head_weights=None)
        model.compile()
        model.fit(gen)
        pickle.dump(model, open("pickle/model.p", "wb"))

    #model.evaluate()
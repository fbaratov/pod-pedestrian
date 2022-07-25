from prep_dataset import retrieve_splits
from trainer import *


def visualize_predictions(model_type, model, pred_set, save_dir=None, show_results=True, score_thresh=.5, nms=.45,
                          samples=0, std_thresh=0):
    if model_type == 'deterministic':
        model = make_deterministic(model)
        draw_set = predict_ssd(model, pred_set, threshold=0.5, nms=0.5, samples=0)
    elif model_type == 'stochastic':
        draw_set = predict_ssd(model, pred_set, threshold=score_thresh, nms=nms, samples=samples, std_thresh=std_thresh)
    else:
        raise ValueError("Incorrect model type!")

    draw_predictions(draw_set, show_results=show_results, save_dir=save_dir, display_std=(model_type == "stochastic"))


if __name__ == "__main__":
    _, _, test_set = retrieve_splits("pickle/caltech_split699")
    model = load_from_path("models/caltech_model_0")

    pred_set = [d["image"] for d in test_set][1515:1521]

    visualize_predictions("stochastic",
                          model,
                          pred_set,
                          save_dir=None,
                          show_results=True,
                          samples=5)

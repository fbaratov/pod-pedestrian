from random import sample

from prep_dataset import retrieve_splits
from trainer import Trainer, DropoutTrainer
import argparse


def test_baseline(model_name="dropout_model_full_0", split_name="full_set", k=100, show_truths=True, score_thresh=.3,
                  nms=.6, show_results=False, save_results=False, eval_map=False):
    trainer = Trainer(model=model_name,
                      splits=retrieve_splits(split_name))

    # generate predictions
    if show_results or save_results:
        trainer.draw_results(k, show_truths, score_thresh, nms, show_results, save_results)

    if eval_map:
        map_score = trainer.evaluate(score_thresh, nms)
        print(map_score)

def test_dropout(model_name="dropout_model_full_0", split_name="full_set", k=100, show_truths=True, score_thresh=.3,
                 nms=.6, show_results=False, save_results=False, eval_map=False):
    trainer = DropoutTrainer(model=model_name,
                             splits=retrieve_splits(split_name))

    # generate predictions
    if show_results or save_results:
        trainer.draw_results(k, show_truths, score_thresh, nms, show_results, save_results)

    if eval_map:
        map_score = trainer.evaluate(score_thresh, nms)
        print(map_score)


def compare_models(model0, model1, split_name, k=100, show_truths=True, score_thresh=.3,
                   nms=.6, show_results=False, save_results=False):
    name0, type0 = model0
    name1, type1 = model1

    if type0 == 'baseline':
        trainer0 = Trainer(model=name0,
                           splits=retrieve_splits(split_name))
    else:
        trainer0 = DropoutTrainer(model=name0,
                                  splits=retrieve_splits(split_name))

    predict_sample = sample(trainer0.d_test, k=k)

    if type1 == 'baseline':
        trainer1 = Trainer(model=name1,
                           splits=retrieve_splits(split_name))
    else:
        trainer1 = DropoutTrainer(model=name1,
                                  splits=retrieve_splits(split_name))

    # mark as comparison by altering model name
    trainer0.model_name = "compare_" + trainer0.model_name
    trainer1.model_name = "compare_" + trainer1.model_name

    # generate predictions
    if show_results or save_results:
        trainer0.draw_results(k, show_truths, score_thresh, nms, show_results, save_results, draw_set=predict_sample)
        trainer1.draw_results(k, show_truths, score_thresh, nms, show_results, save_results, draw_set=predict_sample)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help='model type, either \'baseline\', \'dropout\'')
    parser.add_argument('model_name', type=str, help='model directory name')
    parser.add_argument('--split_name', type=str, help='split directory name', default='full_set')
    parser.add_argument('--score_threshold', type=float, help='score threshold', default=0.3)
    parser.add_argument('--nms', type=float, help='non-maximum suppression threshold', default=0.6)
    parser.add_argument('--map', type=bool, help='if True, evaluates model mean average precision (mAP)', default=False)
    parser.add_argument('--k', type=int, help='number of results to show', default=100)
    parser.add_argument('--show_truths', type=bool, help='if True, draws ground truths on results', default=False)
    parser.add_argument('--show_results', type=bool, help='if True, displays examples of predictions', default=False)
    parser.add_argument('--save_results', type=bool, help='if True, saves images to predictions folder', default=False)
    parser.add_argument('--compare', type=bool,
                        help='if True, compares two provided models by evaluating them on the same set', default=False)
    parser.add_argument('--second_model_type', type=str,
                        help='second model type for comparison, either \'baseline\', \'dropout\'', default=None)
    parser.add_argument('--second_model_name', type=str, help='second model directory name for comparison',
                        default=None)

    a = parser.parse_args()

    if a.compare:
        valid_types = ['baseline', 'dropout']
        if a.model_type in valid_types and a.second_model_type in valid_types:
            compare_models([a.model_name, a.model_type], [a.second_model_name, a.second_model_type], a.split_name, a.k,
                           a.show_truths, a.score_threshold, a.nms, a.show_results, a.save_results)
        else:
            raise ValueError("Invalid model type provided!")
    elif a.model_type == 'baseline':
        test_baseline(a.model_name, a.split_name, a.k, a.show_truths, a.score_threshold, a.nms, a.show_results,
                      a.save_results, a.map)
    elif a.model_type == 'dropout':
        test_dropout(a.model_name, a.split_name, a.k, a.show_truths, a.score_threshold, a.nms, a.show_results,
                     a.save_results, a.map)
    else:
        raise ValueError("Invalid model type provided!")

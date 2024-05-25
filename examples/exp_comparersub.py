from cornac.experiment import Experiment_Explainers
from cornac.models import MTER, ComparERSub
from cornac.explainer import Exp_ComparERSub
from cornac.metrics_explainer import (
    Metric_Exp_PSPNFNS as PSPN,
    Metric_Exp_DIV as DIV,
    Metric_Exp_FPR as FPR,
)
from cornac.datasets import amazon_toy
from cornac.data.reader import Reader
from cornac.eval_methods import StratifiedSplit
from cornac.data.sentiment import SentimentModality

rating = amazon_toy.load_feedback(fmt="UIRT", reader=Reader(min_user_freq=50))
sentiment_data = amazon_toy.load_sentiment(reader=Reader(min_user_freq=50))

md = SentimentModality(data=sentiment_data)

eval_method = StratifiedSplit(
    data=rating,
    group_by="user",
    chrono=True,
    sentiment=md,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
)

# initialize recommenders and explainers
mter = MTER(
    max_iter=20,
    n_user_factors=8,
    n_item_factors=8,
    n_aspect_factors=8,
    n_opinion_factors=8,
    n_bpr_samples=1000,
    n_element_samples=50,
    lambda_reg=0.1,
    lambda_bpr=10,
    lr=0.5,
)

mter.fit(eval_method.train_set)
params = {
    "G1": mter.G1,
    "G2": mter.G2,
    "G3": mter.G3,
    "U": mter.U,
    "I": mter.I,
    "A": mter.A,
    "O": mter.O,
}

comparersub = ComparERSub(
    max_iter=20,
    n_user_factors=8,
    n_item_factors=8,
    n_aspect_factors=8,
    n_opinion_factors=8,
    n_pair_samples=1000,
    n_bpr_samples=1000,
    n_element_samples=50,
    lambda_reg=0.1,
    lambda_bpr=10,
    lambda_d=10,
    lr=0.5,
    min_common_freq=1,
    min_user_freq=2,
    min_pair_freq=1,
    trainable=True,
    verbose=True,
    init_params=params,
)
exp_comparersub = Exp_ComparERSub(comparersub, eval_method.train_set)


# initialize metrics
pspnfns = PSPN()
fdiv = DIV()
fpr = FPR()
fpr_with_input_as_groundtruth = FPR(ground_truth=sentiment_data)

# initialize experiment
models = [(comparersub, exp_comparersub)]
metrics = [fdiv, fpr_with_input_as_groundtruth, pspnfns]
experiment = Experiment_Explainers(
    eval_method=eval_method,
    models=models,
    metrics=metrics,
    rec_k=10,
    distribution=False,
    feature_k=10,
    eval_train=True,
)
experiment.run()
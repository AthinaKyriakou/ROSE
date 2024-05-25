from cornac.experiment import Experiment_Explainers
from cornac.models import EFM, ComparERObj
from cornac.explainer import Exp_ComparERObj_Mod
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
efm = EFM(
    max_iter=20,
    num_explicit_factors=128,
    num_latent_factors=128,
    num_most_cared_aspects=100,
    rating_scale=5.0,
    alpha=0.9,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    trainable=True,
)
efm.fit(eval_method.train_set)
params = {
        "U1": efm.U1,
        "U2": efm.U2,
        "H1": efm.H1,
        "H2": efm.H2,
        "V": efm.V,
}
comparerobj = ComparERObj(
    max_iter=20,
    num_explicit_factors=128,
    num_latent_factors=128,
    num_most_cared_aspects=20,
    rating_scale=5.0,
    alpha=0.7,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    lambda_d=0.1,
    min_user_freq=2,
    trainable=True,
    verbose=True,
    init_params=params,
)
exp_comparerobj = Exp_ComparERObj_Mod(comparerobj, eval_method.train_set)

# initialize metrics
pspnfns = PSPN()
fdiv = DIV()
fpr = FPR()
fpr_with_input_as_groundtruth = FPR(ground_truth=sentiment_data)

# initialize experiment
models = [(comparerobj, exp_comparerobj)]
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
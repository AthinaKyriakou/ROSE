from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data.reader import Reader
from cornac.experiment.experiment_explainers import Experiment_Explainers
from cornac.models import ALS
from cornac.explainer import Exp_ALS
from cornac.metrics_explainer import (
    Metric_Exp_DIV as DIV,
    Metric_Exp_PGF as PGF,
    Metric_Exp_MEP as MEP,
    Metric_Exp_EnDCG as EnDCG,
)

# Load MovieLens
data = movielens.load_feedback(variant="100K", reader=Reader(min_user_freq=150))

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=data, test_size=0.2, exclude_unknowns=False, verbose=True
)

# initialize recommenders and explainers
als = ALS(k=10, max_iter=500, lambda_reg=0.001, alpha=1, verbose=True, seed=6)
als_exp = Exp_ALS(rec_model=als, dataset=ratio_split.train_set)

# initialize metrics
fdiv = DIV()
pgf = PGF()
mep = MEP()
endcg = EnDCG()

# initialize experiment
models = [(als, als_exp)]
metrics = [fdiv, pgf, mep, endcg]
experiment = Experiment_Explainers(
    eval_method=ratio_split,
    models=models,
    metrics=metrics,
    distribution=False,
    rec_k=10,
    feature_k=10,
    eval_train=True,
)
experiment.run()
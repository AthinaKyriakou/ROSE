from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import Reader
from cornac.models import MF
from cornac.explainer import Exp_LIRE
from cornac.experiment import Experiment_Explainers
from cornac.metrics_explainer import Metric_Exp_DIV, Metric_Exp_PGF

mf = MF(k=100, max_iter=100, learning_rate=0.01, lambda_reg=0.001, verbose=True, seed=42)
data = movielens.load_feedback(variant="100K", reader=Reader(min_user_freq=250))
ratio_split = RatioSplit(data, test_size=0.2, seed=42, verbose=True)

exp = Exp_LIRE(mf, ratio_split.train_set)

pgf = Metric_Exp_PGF()
div = Metric_Exp_DIV()

models = [(mf, exp)]
metrics = [pgf, div]
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
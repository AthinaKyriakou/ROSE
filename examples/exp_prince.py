from cornac.datasets import amazon_office as office
from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.models import C2PF
from cornac.explainer import Exp_PRINCE
from cornac.experiment.experiment_explainers import Experiment_Explainers
from cornac.metrics_explainer import Metric_Exp_DIV


ratings = office.load_feedback()
contexts = office.load_graph()

item_graph_modality = GraphModality(data=contexts)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=3.5,
    exclude_unknowns=True,
    verbose=True,
    item_graph=item_graph_modality,
)

c2pf = C2PF(k=100, max_iter=80, variant="c2pf")

prince = Exp_PRINCE(c2pf, ratio_split.train_set, rec_k=10)

fdiv = Metric_Exp_DIV()

# initialize experiment
models = [(c2pf, prince)]
metrics = [fdiv]
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
from cornac.models import ALS
from cornac.explainer import ALSExplainer
from cornac.datasets.goodreads import prepare_data
from cornac.metrics_explainer import PGF, DIV
from cornac.metrics_explainer import Explainers_Experiment


# Load dataset
dataset_dense = prepare_data(
    data_name="goodreads_uir_1000", test_size=0, verbose=True, sample_size=1, dense=True
)
assert dataset_dense is not None

# Recommendation model
als = ALS(
    k=10, 
    max_iter=500, 
    lambda_reg=0.001, 
    alpha=1, 
    verbose=True, 
    seed=6
)

# Explainer
explainer = ALSExplainer(als, dataset_dense.train_set)
als_als = (als, explainer)

# Metrics for explainer
pgf = PGF(phi=10)
fdiv = DIV()

# Experiment
experiment = Explainers_Experiment(
    eval_method=dataset_dense,
    models=[als_als],
    metrics=[pgf, fdiv],
    rec_k=10,
    feature_k=10,
    eval_train=True,
    distribution=True,
)
experiment.run()

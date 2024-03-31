from cornac.models import EMF, NEMF
from cornac.explainer import EMFExplainer
from cornac.datasets.goodreads import prepare_data
from cornac.metrics_explainer import MEP, EnDCG, PGF, DIV
from cornac.metrics_explainer import Explainers_Experiment
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
# Load dataset
dataset_dense = prepare_data(
    data_name="goodreads_uir_1000", test_size=0, verbose=True, sample_size=1, dense=True
)
assert dataset_dense is not None

# Recommendation models
emf = EMF(
    k=10,
    max_iter=500,
    learning_rate=0.001,
    lambda_reg=0.1,
    explain_reg=0.01,
    verbose=True,
    seed=6,
    num_threads=6,
    early_stop=True,
)
nemf = NEMF(
    k=10, 
    max_iter=500, 
    learning_rate=0.001, 
    lambda_reg=0.1, 
    explain_reg=0.01, 
    novel_reg=1, 
    verbose=True, 
    seed=6, 
    num_threads=6, 
    early_stop=True
)

# Explainer
emf_emf = (emf, EMFExplainer(emf, dataset_dense.train_set))
nemf_emf = (nemf, EMFExplainer(nemf, dataset_dense.train_set))

# Metrics for explainer
mep = MEP()
endcg = EnDCG()
pgf = PGF(phi=10)
fdiv = DIV()

# Experiment
experiment = Explainers_Experiment(
    eval_method=dataset_dense,
    models=[emf_emf, nemf_emf],
    metrics=[mep, endcg, pgf, fdiv],
    rec_k=10,
    feature_k=10,
    eval_train=True,
    distribution=True,
)
experiment.run()

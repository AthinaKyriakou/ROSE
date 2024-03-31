from cornac.metrics_explainer import Explainers_Experiment
from cornac.models import MF, EMF, NEMF, ALS
from cornac.explainer import EMFExplainer, ALSExplainer, PHI4MFExplainer
from cornac.datasets.goodreads import prepare_data
from cornac.metrics_explainer import MEP, EnDCG, PGF

# Load dataset
dataset = prepare_data(data_name="goodreads_uir",test_size=0, verbose=True, sample_size=1, dense=True)
assert dataset is not None

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
mf = MF(
    k=10, 
    max_iter=500, 
    learning_rate=0.001, 
    lambda_reg=0.1, 
    verbose=True, 
    seed=6
)

# Explainer
mf_phi = (mf, PHI4MFExplainer(mf, dataset.train_set))
print("mf_phi done")
emf_phi = (emf, PHI4MFExplainer(emf, dataset.train_set))
print("emf_phi done")
nemf_phi = (nemf, PHI4MFExplainer(nemf, dataset.train_set))    
print("nemf_phi done")

# Metrics for explainer
mep = MEP()
endcg = EnDCG()
pgf = PGF(phi=10)

# Experiment
experiment = Explainers_Experiment(
    eval_method=dataset, 
    models=[mf_phi, emf_phi, nemf_phi], 
    metrics=[mep, endcg, pgf], 
    rec_k=10, 
    feature_k=10, 
    eval_train=True, 
    distribution=True
)
experiment.run()
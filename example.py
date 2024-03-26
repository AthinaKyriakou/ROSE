import pandas as pd
import cornac

from cornac.eval_methods import RatioSplit
from cornac.models import EMF, NEMF, ALS
from cornac.explainer import EMFExplainer, ALSExplainer
from cornac.datasets.goodreads import prepare_data
from cornac.metrics_explainer import Explainers_Experiment, MEP, EnDCG, PGF

VERBOSE = False
SEED = 42

# dataset
dataset_dense = prepare_data(data_name="goodreads_uir_1000",test_size=0, verbose=True, sample_size=1, dense=True)

amz_clth = cornac.datasets.amazon_clothing.load_feedback()
ac = RatioSplit(data=amz_clth, test_size=0.2, seed=SEED, verbose=True, exclude_unknowns=True)

# recommender models
emf = EMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, verbose=True, seed=6, num_threads=6, early_stop=True)
nemf = NEMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, novel_reg=1, verbose=True, seed=6, num_threads=6, early_stop=True)
als = ALS(k=10, max_iter=500, lambda_reg=0.001, alpha=1, verbose=True, seed=6)

# (recommender, explainer) pairs
emf_emf = (emf, EMFExplainer(emf, ac.train_set))
nemf_emf = (nemf, EMFExplainer(nemf, ac.train_set))
als_als = (als, ALSExplainer(als, ac.train_set))

# metrics
mep = MEP()
endcg = EnDCG()
pgf = PGF(phi=10)

# experiment
experiment = Explainers_Experiment(eval_method=ac, 
                                    models=[emf_emf, nemf_emf, als_als], 
                                    metrics=[mep, endcg, pgf], 
                                    rec_k=10, 
                                    feature_k=10, 
                                    eval_train=True, 
                                    distribution=True)
experiment.run()
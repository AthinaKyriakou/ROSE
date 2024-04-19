import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cornac
# from cornac.data import Reader, SentimentModality, FeatureModality
from cornac.eval_methods import RatioSplit
from cornac.metrics import RMSE, MAE, AUC, NDCG, FMeasure
# from cornac.hyperopt import Discrete, Continuous
# from cornac.hyperopt import GridSearch
from cornac import Experiment

from cornac.metrics_explainer import Explainers_Experiment
from cornac.models import MF, EMF, NEMF, ALS, MTER, FMRec, NMF, EFM
from cornac.explainer import Exp_EMF, Exp_EFM, Exp_ALS, Exp_PHI4MF, Exp_MTER, Exp_EFM_Mod, Exp_LIMERS
from cornac.datasets.goodreads import prepare_data
# from cornac.utils.libffm_mod import LibffmModConverter
from cornac.metrics_explainer import Metric_Exp_MEP, Metric_Exp_EnDCG, Metric_Exp_PGF, Metric_Exp_DIV, Metric_Exp_FPR, Metric_Exp_FA, Metric_Exp_RA, Metric_Exp_PSPNFNS
from cornac.data import Reader, SentimentModality

# TODO: Add hyperparam tuning
# TODO: Add more metrics for explainers

VERBOSE = False
SEED = 42

ground_truth_path = "cornac/datasets/good_reads/goodreads_sentiment_full.txt"
sentiment_fpath = 'tests/dataset/goodreads_sentiment.txt'
sentiment = Reader().read(sentiment_fpath, fmt='UITup', sep=',', tup_sep=':')
rating_fpath = 'tests/dataset/goodreads_rating.txt'
rating = Reader(min_item_freq=20).read(rating_fpath, fmt='UIR', sep=',')
sentiment_modality = SentimentModality(data=sentiment)

# Load Goodreads Sentiment Data
gr_sentiment = RatioSplit(data=rating, test_size=0.2, exclude_unknowns=True, sentiment=sentiment_modality, verbose=VERBOSE, seed=SEED)

# Load Movielens
ml_1M = cornac.datasets.movielens.load_feedback(variant='1M')
ml = RatioSplit(data=ml_1M, test_size=0.2, rating_threshold=4.0, seed=SEED, verbose=True)

# Load Netflix
nflix = cornac.datasets.netflix.load_feedback(variant='small')
nf = RatioSplit(data=nflix, test_size=0.2, seed=SEED, verbose=True)

# Load Amazon Toys
amz_toy = cornac.datasets.amazon_toy.load_feedback()
amz_toy_sentiment = cornac.datasets.amazon_toy.load_sentiment()
amz_toy_sentiment_modality = SentimentModality(data = amz_toy_sentiment)

at = RatioSplit(data=amz_toy, test_size=0.2, seed=SEED, verbose=True)
at_sentiment = rs = RatioSplit(data=amz_toy, test_size=0.2, exclude_unknowns=True, sentiment=amz_toy_sentiment_modality, verbose=True, seed=SEED)

# Load Amazon Clothing
amz_clth = cornac.datasets.amazon_clothing.load_feedback()
ac = RatioSplit(data=amz_clth, test_size=0.2, seed=SEED, verbose=True)

# goodreads data sparse
gr_interact = prepare_data(data_name="goodreads_uir",test_size=0.2, verbose=True, sample_size=1, dense=False)

# goodreads data limers
gr_limers = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=False, item=True, user=False, sample_size=1, seed=SEED)

datasets = {
    "Amazon Clothing Interactions": ac,  
    "Amazon Toys Interactions": at,
    "MovieLens 1M": ml, 
    "Goodreads Interactions": gr_interact,
    "Netflix": nf,
    }

# define metrics to evaluate the models
metrics = [MAE(), RMSE(), NDCG(k=10), AUC(), FMeasure(k=10)]
# fpr_ground_truth = FPR(fpath=ground_truth_path)
expl_metrics = [Metric_Exp_MEP(), Metric_Exp_EnDCG(), Metric_Exp_PGF(phi=10), Metric_Exp_FA(), Metric_Exp_RA(), Metric_Exp_DIV()]  # fpr_ground_truth

# loop for MF-explanation methods
for d_name, data in datasets.items():
    print(d_name)
    # initialize models to compare
    mf = MF(k=10, max_iter=200, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=SEED)
    emf = EMF(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, verbose=True, seed=SEED, num_threads=6, early_stop=True)
    emf_p = EMF(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, verbose=True, seed=SEED, num_threads=6, early_stop=True)
    nemf = NEMF(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, novel_reg=1, verbose=True, seed=SEED, num_threads=6, early_stop=True)
    nemf_p = NEMF(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, novel_reg=1, verbose=True, seed=SEED, num_threads=6, early_stop=True)
    als = ALS(k=10, max_iter=200, lambda_reg=0.001, alpha=1, verbose=True, seed=SEED)
    models = [emf, nemf, als, mf, emf_p, nemf_p]
    
    # Recommender Performance
    Experiment(eval_method=data, models=models, metrics=metrics, save_dir="output/" + d_name +  "/recsys_eval/").run()
    print("RecSys evaluation done")
    
    # Match Recommenders with Explainers
    emf_emf = (emf, Exp_EMF(emf, data.train_set))
    Explainers_Experiment(eval_method=data, models=[emf_emf], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("EMF_EMF done")
    del emf, emf_emf

    nemf_emf = (nemf, Exp_EMF(nemf, data.train_set))
    Explainers_Experiment(eval_method=data, models=[nemf_emf], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("NEMF_EMF done")
    del nemf, nemf_emf

    print("starting ALS explainer")
    als_als = (als, Exp_ALS(als, data.train_set))
    Explainers_Experiment(eval_method=data, models=[als_als], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("ALS_ALS done")
    del als, als_als

    mf_phi = (mf, Exp_PHI4MF(mf, data.train_set))
    Explainers_Experiment(eval_method=data, models=[mf_phi], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("MF_PHI done")
    del mf, mf_phi

    emf_phi = (emf_p, Exp_PHI4MF(emf_p, data.train_set))
    Explainers_Experiment(eval_method=data, models=[emf_phi], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("EMF_PHI done")
    del emf_p, emf_phi

    nemf_phi = (nemf_p, Exp_PHI4MF(nemf_p, data.train_set))
    Explainers_Experiment(eval_method=data, models=[nemf_phi], 
                          metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/explanation_eval/").run()
    print("NEMF_PHI done")
    del nemf_p, nemf_phi
    # expl_models = [emf_emf, nemf_emf, als_als, mf_phi, emf_phi, nemf_phi]

    print("Explanation evaluation done for dataset ", d_name)

   
    # Explainer Performance
    # Explainers_Experiment(eval_method=data, models=expl_models, 
    #                       metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/" + d_name + "/").run()
    


# stuff that needs to be evaluated separately due to data incompatabilty (until I figure out sth else)
fm = FMRec(num_iter=200)
fm_LimeRS = Exp_LIMERS(fm, gr_limers.train_set)
print("FM_LIMERS fit done")
Experiment(eval_method=gr_limers, models=[fm], metrics=metrics, save_dir="output/goodreads_limers/recsys_eval/").run()
print("RecSys evaluation done")
# Explainer Performance
Explainers_Experiment(eval_method=gr_limers, models=[(fm,fm_LimeRS)], 
                      metrics=expl_metrics, rec_k=10, feature_k=10, eval_train=True, distribution=True, save_dir="output/goodreads_limers/").run()
print("Explanation evaluation FM_LIMERS done")

datasets = {
            "goodreads_sentiment": gr_sentiment,
            "amazon_toys_sentiment": at_sentiment
            }

for d_name,data in datasets.items():
    print('\n')
    print(d_name)
    print('\n')
    efm = EFM(num_explicit_factors = 40, num_latent_factors = 60, num_most_cared_aspects = 15, rating_scale = 5.0, alpha = 0.85, lambda_x = 1, lambda_y = 1, lambda_u = 0.01, lambda_h = 0.01, lambda_v = 0.01, max_iter = 100, verbose = VERBOSE, seed = SEED)
    efm.fit(data.train_set)
    efm_exp = Exp_EFM(efm, efm.train_set)
    print("efm_exp done")
    # modified EFM Explainer
    efm_mod = EFM(num_explicit_factors = 40, num_latent_factors = 60, num_most_cared_aspects = 15, rating_scale = 5.0, alpha = 0.85, lambda_x = 1, lambda_y = 1, lambda_u = 0.01, lambda_h = 0.01, lambda_v = 0.01, max_iter = 100, verbose = VERBOSE, seed = SEED)
    efm_mod.fit(data.train_set)
    mod_efm_exp = Exp_EFM_Mod(efm_mod, efm_mod.train_set)
    print("mod_efm_exp done")
    # Experiment(eval_method=data, models=[efm, efm_mod], metrics=metrics, save_dir="output/" + d_name +  "/recsys_eval/").run()
    print('start evaluating explainers')
    Explainers_Experiment(eval_method=data, models=[(efm, efm_exp), (efm_mod, mod_efm_exp)], metrics=expl_metrics, save_dir="output/" + d_name +  "/").run()
    del efm_mod, mod_efm_exp, efm, efm_exp 

# TODO: see if I need to run explainer experiment for MFs with tuples as models-input as well
# TODO: Add metrics compatible for sentiment explainers
#MTER
# TODO: evaluate separately
# mter = MTER(max_iter=200, n_aspect_factors=8, n_item_factors=5, n_opinion_factors= 5, n_user_factors= 10, lambda_bpr=10, lambda_reg= 10, n_bpr_samples=1000, n_element_samples=50)
# mter.fit(data.train_set)
# mter_mter = MTERExplainer(mter, mter.train_set)
# print("MTER_MTER done")
# Explainers_Experiment(eval_method=data, models=[(mter, mter_mter)], metrics=expl_metrics, save_dir="output/" + d_name +  "/").run()
# del mter, mter_mter, 

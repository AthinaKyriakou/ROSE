import unittest
import numpy as np
import pandas as pd

from cornac.datasets.goodreads import prepare_data
from cornac.models import FMRec, EFM, MTER, ALS, MF, EMF, NEMF
from cornac.explainer import Exp_LIMERS, Exp_EFM, Exp_MTER, Exp_SU4EMF, Exp_ALS, Exp_PHI4MF
from cornac.metrics_explainer import Metric_Exp_PSPNFNS
from cornac.metrics_explainer import Metric_Exp_DIV as DIV
from cornac.metrics_explainer import Metric_Exp_MEP as MEP
from cornac.metrics_explainer import Metric_Exp_FPR as FPR
from cornac.metrics_explainer.exp_experiment import Explainers_Experiment
from config import cfg

class TestExperiment(unittest.TestCase):

    def test_valid_limers_metric(self):
        """test experiment can run with no issue for limers"""
        rs = prepare_data(data_name="goodreads_limers", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        fm = FMRec()
        limers = Exp_LIMERS(rec_model=fm, dataset=rs.train_set)
        metrics = [Metric_Exp_PSPNFNS(), FPR()]
        results = Explainers_Experiment(eval_method=rs, models=[(fm, limers)], metrics=metrics,rec_k=2, feature_k=2, eval_train=True).run()

    def test_valid_sentiment_metric(self):
        """test experiment can run with no issue for sentiment"""
        rs = prepare_data(data_name="goodreads", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        efm = EFM()
        efm_exp = Exp_EFM(rec_model=efm, dataset=rs.train_set)
        mter = MTER()
        mter_exp = Exp_MTER(rec_model=mter, dataset=rs.train_set)
        metrics = [Metric_Exp_PSPNFNS(), FPR()]
        results = Explainers_Experiment(eval_method=rs, models=[(efm, efm_exp), (mter, mter_exp)], metrics=metrics,rec_k=2, feature_k=2, eval_train=True).run()

    def test_valid_mf_metric(self):
        """test experiment can run with no issue for mf models"""
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        emf = EMF()
        als = ALS()
        mf = MF()
        nemf = NEMF()
        emf_exp = Exp_SU4EMF(rec_model=emf, dataset=rs.train_set)
        als_exp = Exp_ALS(rec_model=als, dataset=rs.train_set)
        phi4 = Exp_PHI4MF(rec_model=nemf, dataset=rs.train_set)
        models = [(emf, emf_exp), (als, als_exp), (nemf,phi4)]
        metrics = [MEP()]
        results = Explainers_Experiment(eval_method=rs, models=models, metrics=metrics,rec_k=2, feature_k=2, eval_train=True).run()

    def test_validate_models(self):
        """validate invalid models raise ValueError"""
        rs = prepare_data(data_name="goodreads_uir_1000", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        metrics = [MEP()]
        try:
            Explainers_Experiment(rs, None, metrics)
        except ValueError:
            assert True

    def test_validate_metrics(self):
        """validate invalid metrics raise ValueError"""
        rs = prepare_data(data_name="goodreads", test_size=0.2, dense=True, item=True, user=True, sample_size=0.1, seed=21)
        emf = EMF()
        emf_exp = Exp_SU4EMF(rec_model=emf, dataset=rs.train_set)
        models = [emf, emf_exp]
        try:
            Explainers_Experiment(rs, models, None)
        except ValueError:
            assert True


if __name__ == '__main__':
    unittest.main()
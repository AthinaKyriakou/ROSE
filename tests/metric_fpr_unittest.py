import unittest
import numpy as np
import pandas as pd
from config import cfg

from cornac.models import EFM, FMRec 
from cornac.explainer import EFMExplainer, LimeRSExplainer
from cornac.metrics_explainer import FPR
from cornac.datasets.goodreads import prepare_data

class TestFPR(unittest.TestCase):
    """
        Test for FPR
        Can work with Recommender-Explainer pairs:
            - FMRec-LimeRSExplainer
            - EFM-EFMExplainer
            - MTER-MTERExplainer
            
        Not work with:
        - ALS-ALSExplainer
            - EMF-EMFExplainer
            - NEMF-EMFExplainer
            - MF-PHI4MFExplainer
            - EMF-PHI4MFExplainer
            - NEMF-PHI4MFExplainer
            
        In this test, we use EFM-EFMExplainer and FMRec-LIMERS as an example

    """
    def test_FPR_EFM(self):
        """
        Test for FPR with recommender cornac.models.EFM and explainer cornac.explainer.EFMExplainer
        FPR can work on MTER-MTERExplainer similar to EFM-EFMExplainer
        """
        # Prepare the dataset
        print("Start testing FeaturePrecisionRecall")
        rs = prepare_data(data_name="goodreads", test_size=0, sample_size=0.1, dense=True)
        # init the recommendation model
        efm = EFM()
        efm = efm.fit(rs.train_set)
        # Init the explainer
        efm_exp = EFMExplainer(efm, rs.train_set)

        fpr = FPR()
        # For MTER, we need to pass in the path of the sentiment data,
        # fpr = FPR(sentiment_path="./dataset/goodreads_sentiment.txt")
        [fp, fr, ff1], [fp_list, fr_list, ff1_list] = fpr.compute(efm, efm_exp)

        assert fp >= 0.0
        assert fr >= 0.0
        assert ff1 >= 0.0
        # each explanation has a precision, recall, and f1 value
        assert len(fp_list) == len(ff1_list)
        assert len(fr_list) == len(ff1_list)
        
    def test_FPR_LIMERS(self):
        """
        Test for FPR with recommender cornac.models.FMRec and explainer cornac.explainer.LimeRSExplainer
        """
        rs = prepare_data(data_name="goodreads_limers", test_size=0, verbose=False, sample_size=1, dense=True)
        fm = FMRec(verbose=False).fit(rs.train_set)
        limers = LimeRSExplainer(fm, rs.train_set)
        
        fpr = FPR()
        [fp, fr, ff1], [fp_list, fr_list, ff1_list] = fpr.compute(fm, limers)

        assert fp >= 0.0
        assert fr >= 0.0
        assert ff1 >= 0.0
        
        assert len(fp_list) == len(ff1_list)
        assert len(fr_list) == len(ff1_list)
            
        
if __name__ == '__main__':
    unittest.main()
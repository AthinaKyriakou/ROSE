Quick Start
===========

.. toctree::
   :maxdepth: 2


Experiment Explainer
--------------------
After ROSE is installed, you can use the following command to run your experiment for recommenders and explainers:

.. code-block:: python

   from cornac.datasets import movielens
   from cornac.eval_methods import RatioSplit
   from cornac.experiment import Experiment_Explainers
   from cornac.models import EMF, NEMF, ALS
   from cornac.explainer import Exp_ALS, Exp_SU4EMF
   from cornac.metrics_explainer import (
      Metric_Exp_DIV as DIV,
      Metric_Exp_PGF as PGF,
      Metric_Exp_MEP as MEP,
      Metric_Exp_EnDCG as EnDCG,
   )

   # Load MovieLens
   data = movielens.load_feedback(variant="100K")

   # Define an evaluation method to split feedback into train and test sets
   ratio_split = RatioSplit(data=data, test_size=0.2, exclude_unknowns=False, verbose=True)

   # initialize recommenders and explainers
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
      early_stop=True,
   )
   als = ALS(k=10, max_iter=500, lambda_reg=0.001, alpha=1, verbose=True, seed=6)
   als_exp = Exp_ALS(rec_model=als, dataset=ratio_split.train_set)
   emf_exp = Exp_SU4EMF(rec_model=emf, dataset=ratio_split.train_set)
   nemf_exp = Exp_SU4EMF(rec_model=nemf, dataset=ratio_split.train_set)

   # initialize metrics
   fdiv = DIV()
   pgf = PGF()
   mep = MEP()
   endcg = EnDCG()

   # initialize experiment
   models = [(emf, emf_exp), (als, als_exp), (nemf, nemf_exp)]
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



More examples can be found in `demo/experiment_explainers_demo.ipynb <https://github.com/AthinaKyriakou/ROSE/blob/main/demo/experiment_explainers_demo.ipynb>`_


Visualization
-------------
Examples can be found in `demo/visualization_demo.ipynb <https://github.com/AthinaKyriakou/ROSE/blob/main/demo/visualization_demo.ipynb>`_


Lexicon Generation
------------------
Examples can be found in `demo/genetate_lexicon_demo.ipynb <https://github.com/AthinaKyriakou/ROSE/blob/main/demo/genetate_lexicon_demo.ipynb>`_


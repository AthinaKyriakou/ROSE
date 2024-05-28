# ROSE: Towards A Framework for Recommender Systems' Explainability

To facilitate the usage of explainability approaches for recommender systems, systematize their benchmarking, and enforce reproducibility we propose ROSE, a unified open-source framework to generate explanations and evaluate explainable recommender systems and suitable post-hoc explanation approaches
![ROSE Main Interaction Workflow](/rose_pipeline.png)

## Installation
ROSE supports `Python >= 3.9`. We highly recommend using a Python virtual environment to install the packages as some of them (particularly Cython and Sklearn) are only compatible in certain versions. Create a virtual environment within the project's repository (i.e., in /ROSE/). Within the project's repository and with the virtual environment activated run in a terminal:
``` sh
bash setup.sh
```
And then build the project:
```sh
python setup.py install
python setup.py build_ext --inplace
```

## Quick Start
The Explainers_Experiment needs recommenders, explainers and metrics. Here is one example that you can run by executing `python example.py`:
  
``` python
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
```

## Resources
* Tutorials on adding a new explainability method `tutorials/add_explainer.py` and metric `tutorials/add_metric_explainer.py`
* Demos on creating a lexicon, running explainability experiment and generating visualizations `demo/`
* Code examples on how to generate explanations based on all the integrated explainable recommendation algorithms and post-hoc methods `examples/`.

## Cite
TBA

## The Team
This code base would not have been possible without the work of amazing contributors, listed here in alphabetical order: Athina Kyriakou, Nan Li, Yingying Liu, Konstantina Timoleon, Kathrin Wardatzky, Krzysztof Wroblewski, and Yiqin Zhang.

## License
[Apache 2.0](LICENSE)
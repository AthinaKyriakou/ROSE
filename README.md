# RecOmmender Systems Explainers

## Installation
We highly recommend using a Python virtual environment to install the packages as some of them (particularly Cython and Sklearn) are only compatible in certain versions. Create a virtual environment within the project's repository (i.e., in /ROSE/). Within the project's repository and with the virtual environment activated run in a terminal:
``` sh
bash setup.sh
```
And then build the project.
```sh
python setup.py install
python setup.py build_ext --inplace
```

## Quick Start
The Explainers_Experiment needs recommenders, explainers and metrics. Here is one example that you can run by executing `python example.py`.
  
``` python
import pandas as pd

from cornac.models import EMF, NEMF, ALS
from cornac.explainer import EMFExplainer, ALSExplainer
from cornac.datasets.goodreads import prepare_data
from cornac.experiment.experiment_explainers import Experiment_Explainers
from cornac.metrics_explainer import MEP, EnDCG, PGF

# dataset
dataset_dense = prepare_data(data_name="goodreads_uir_1000",test_size=0, verbose=True, sample_size=1, dense=True)

# recommender models
emf = EMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, verbose=True, seed=6, num_threads=6, early_stop=True)
nemf = NEMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, novel_reg=1, verbose=True, seed=6, num_threads=6, early_stop=True)
als = ALS(k=10, max_iter=500, lambda_reg=0.001, alpha=1, verbose=True, seed=6)

# (recommender, explainer) pairs
emf_emf = (emf, EMFExplainer(emf, dataset_dense.train_set))
nemf_emf = (nemf, EMFExplainer(nemf, dataset_dense.train_set))
als_als = (als, ALSExplainer(als, dataset_dense.train_set))

# metrics
mep = MEP()
endcg = EnDCG()
pgf = PGF(phi=10)

# experiment
experiment = Experiment_Explainers(eval_method=dataset_dense, 
                                    models=[emf_emf, nemf_emf, als_als], 
                                    metrics=[mep, endcg, pgf], 
                                    rec_k=10, 
                                    feature_k=10, 
                                    eval_train=True, 
                                    distribution=True)
experiment.run()
```

There are more demo for experiments in `demo/metrics_*_demo.ipynb`. Note that only valid (recommender, explainer) pairs can be processed by the pipeline. Furthermore, if one metric is not applicable for a (recommender, explainer) pair, a 'N/A' would be returned in the result. 

## Cite

## The Team

## License
ROSE has an MIT License. All data and code in this project can only be used for academic purposes.

## Acknowledgments

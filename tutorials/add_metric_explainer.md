# Contributing an evaluation metric for explainer to Cornac

This tutorial describes how to add an evaluation metric for explainer into ROSE. We assume that you have already forked the ROSE repository to your own account.

## Adding a metric for explanation

The evaluation metrics for explainer are inside the `metrics_explainer` directory, which is different for the metrics for recommenders. For convenience assume that the metric of interest is PGF.
```
ROSE    
│
└───cornac
│   │
│   └───metrics_explainer
│       │   __init__.py
│       │   metrics.py
│       │
│       └───pgf
│           │   __init__.py
│           │   metric_pgf.py
│           │   requirements.txt
```
Note that you only need to add the `pgf` branch as the rest of the structure is already in place.

### 1. Extending the Metrics class

The starting point is to create a class called ``PGF``, which extends the generic class ``Metrics`` implemented inside `metrics_explainer/metrics.py`. The minimal structure of our new class is as follows:  
```python
from ..metrics import Metrics

class PGF(Metrics):
    """Prediction Gap Fidelity

    Parameters
    ----------

    References
    ----------
    """

    def __init__(self, name="PMF", , rec_k=10, feature_k=10, ...):
        super().__init__(name=name, rec_k=rec_k, feature_k=feature_k)

    def compute(self, **kwargs):
        raise NotImplementedError()
```
And then we need to implement the `compute()` function which should return the computed metric value.

### 2. Making new metric available to ROSE
Include a `pgf/__init__.py` file:
```python
from .metric_pgf import PGF
```

And update `../metrics_explainer/__init__.py` by adding the following line:
```python
from .pgf import PGF
```

### 3. Indicating dependencies

If any external library (e.g., TensorFlow, PyTorch)  is required for your explainer, you can add it into `requirements.txt` and indicate which versions are required. Here is a sample of a `requirements.txt` file:

```
tensorflow>=1.10.0
```

### 4. Adding unit tests

All tests are grouped into the ``tests`` folder, in the root of the repository. And there is a file `test_config.yml` contains some basic parameters for recommenders.


## Using Cython and C/C++ (optional)

If you are interested in using [Cython](https://cython.org/) to implement the algorithmic part of your model,  you will need to declare your Cython extension inside  `./setup.py`.
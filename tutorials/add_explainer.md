# Contributing an explainer to ROSE

This tutorial describes how to integrate an explanation method i.e. explainer into ROSE. We assume that you have already forked the ROSE repository to your own account.

## Directory & file structure

For convenience assume that the explainer of interest is PHI4MF. A new explainer can be added as one python file like follows:
```
ROSE    
│
└───cornac
│   │
│   └───explainer
│       │   __init__.py
│       │   explainer.py
│       │   phi4mf_explainer.py
```
Note that you only need to add the `phi4mf_explainer.py` file as the rest of the structure is already in place.

## Creating a new explainer in 3 steps

### 1. Extending the Explainer class

The main file is `phi4mf_explainer.py`, which will contain your explainer related codes.  Here is the minimal structure of such file:
```python
from .explainer import Explainer

class PHI4MFExplainer(Explainer):
    """Post hoc explanation for Matrix Factorization

    Parameters
    ----------

    References
    ----------
    """

    def __init__(self, rec_model, dataset, name='PHI4MF', ...):
        super().__init__(name=name, rec_model=rec_model, dataset=dataset) 

    def explain_one_recommendation_to_user(self, user_id, item_id, **kwargs):
	    """
        Provide an explanation for one user and one item
        user_id: one user
        	The index of the user
        item_id: one item
        	The index of the item which is recommended to the user and needed to be explained.
        returns
        -------
        Explanation for this recommendation
        """
```
Every model extends the generic class `Explainer`. All you need to do is redefine the functions `explain_one_recommendation_to_user` listed in the above `phi4mf_explainer.py` file.  

The `explain_one_recommendation_to_user` function is called by another function named `explain_recommendations`, which is used for experiments and other methods.

### 2. Making your explainer available to ROSE
Update `./cornac/explainer/__init__.py` by adding the following line:
```python
from .phi4mf_explainer import PHI4MFExplainer
```

### 3. Indicating dependencies
If any external library (e.g., TensorFlow, PyTorch)  is required for your explainer, you can add it into `requirements.txt` and indicate which versions are required. Here is a sample of a `requirements.txt` file:

```
tensorflow>=1.10.0
```


## Summary

In short, add a new explainer to ROSE involves,

- Creating new file
    - [x] ./cornac/explainer/name_explainer.py
- Implementing function
     - [x] Explainer.explain_one_recommendation_to_user()
- Updating existing files
     - [x] ./cornac/models/\_\_init__.py
     

  
## Using Cython and C/C++ (optional)

If you are interested in using [Cython](https://cython.org/) to implement the algorithmic part of your model,  you will need to declare your Cython extension inside  `./setup.py`.
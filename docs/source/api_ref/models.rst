Models
======


Below are the models that ROSE added. 

Other models can be found in the original Cornac documentation `here <https://cornac.readthedocs.io/en/v2.0.0/api_ref/models.html>`_.

Recommender(Generic Class)
--------------------------
We add a new method to the generic class `Recommender` to recommend to multiple users at once. This function is used for Explainer and Experiment_Explainers.

.. autoclass:: cornac.models.Recommender
   :members: recommend_to_multiple_users

Alternating Least Squares for Implicit Datasets (ALS)
-----------------------------------------------------
.. automodule:: cornac.models.als.recom_als
   :members:

Explainable Matrix Factorization (EMF)
--------------------------------------
.. automodule:: cornac.models.emf.recom_emf
   :members:

Novel and Explainable Matrix Factorisation (NEMF)
-------------------------------------------------
.. automodule:: cornac.models.nemf.recom_nemf
   :members:

Factoriazation Machine Recommender Algorithm (FM_py)
-----------------------------------------------------
.. automodule:: cornac.models.fm_py.recom_fm_py
   :members:



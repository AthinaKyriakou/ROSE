This file serves as a guide for the ProtoMF model, users can refer to this file in order to understand the procedure of the model

# References
Melchiorre, A.B., Rekabsaz, N., Ganhör, C. and Schedl, M.,
    2022, September. Protomf: Prototype-based matrix factorization for effective and explainable
    recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 246-256).

# Overview of the model
ProtoMF uses prototypes to represent users or items. These prototypes help to increase the recommendations reasoning as they are interpretable for everyone. The procedure of predictions consists of different main parts, namely, the creation of prototypes which can subsequently be used for user/item embedding and  the computation part in which the model gets trained on training data. ProtoMF can be implemented in 3 different variations, one being a model which represents user in prototypes, one which represents items as prototypes and lastly the combined model which represents both in prototypes.
For all models, the first step would be the extraction of prototypes repre- sented in vectors with a specified dimension for either users, items or both. After the creation of these prototypes, we use so-called feature extractors to transform each user or item or both into vectors displaying the similarity to each of the prototypes. For determining the similarity, ProtoMF applies the shifted cosine similarity to the two vectors. During the training procedure, we take these newly created user and item vectors to compute the relation between a user and an item, this relation is displayed in the U-score, I-score or UI-score, depending on which model was chosen. These scores are computed as the dot product of the the embedded user/item vectors.


# Files
* dataset.py - an extra class for the management of datasets, this class takes care of negative sampling
* evaluator.py - this class takes care of evaluation processes with the chosen metrics
* explainer_fuctions.py - this file contains functions that can be utilized to analyze prototype
* feature_extractor.py - this file builds feature extractors for all kind of variations, depending on the selected model
* protoMF.py - here, we desribe the instance of which the recommender models are built upon. The ProtoMF model is based on pytorch and implements a forward function which is used for predicting the interaction between user and item
* recom_protomf.py - entry point to the model, from here we build a model, train it and evaluate it
* standard_params.py - here we define the standard parameters used in this model this ensures easy usability for experiments regarding these parameters

# Functions of the main file recom_protomf.py in detail
* _build_model - This function builds a model for the specified variations

* fit - This function first builds the model by calling the above function, defines all necessary elements such as the optimizer and the dataloaders. Once everything is prepared, the function iterates over the epochs and batches to train the model. The function preliminary terminates if no improvement happen over a certain number of epochs (max_patience). During the training procedure, we produce predictions, evaluate them and subsequently take a step by executing a backward pass.
After each epoch, we evaluate the current model and save the results.
For explanation purposes, we save matrices after training the models. If user prototypes were produced, we save the embedding matrix (displaying relationship between items and user-prototypes). If item prototypes were produces, we save a matrix displaying the relationship between items and item-prototypes.

* score - This function is responsible to make a prediction for a user given a list of items for which predictions should be generated.

* compute_metrics - This function computes the metrics for validation purposes

* recommend  - This function takes a list of users and returns the n best recommendations for each. Unlike other models, this recommender already generates explanation while generating recommendations, hence we also include the explanation for each recommendation.

* This file furthermore contains three functions computing different losses, namely: bce_loss, bpr_loss and sampled_softmax_loss
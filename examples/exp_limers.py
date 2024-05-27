from cornac.datasets import amazon_toy
import numpy as np
from cornac.data import FeatureModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment_Explainers
from cornac.models import FMRec
from cornac.explainer import Exp_LIMERS
from cornac.metrics_explainer import (
    Metric_Exp_PSPNFNS as PSPN,
    Metric_Exp_DIV as DIV,
    Metric_Exp_FPR as FPR,
)


def create_item_features_from_aspects(at_sentiment):
    """Separate aspects and opinions from sentiment data and create item and user features from them."""
    items = {}
    users = {}
    for _, row in enumerate(at_sentiment):
        user, item, sentiments = row
        if user not in users:
            users[user] = []
        if item not in items:
            items[item] = []
        for sentiment in sentiments:
            if sentiment[0] not in items[item]:
                items[item].append(sentiment[0])  # aspect adds to item feature
            if sentiment[1] not in users[user]:
                users[user].append(sentiment[1])  # opinion adds to user feature

    item_aspect_pairs = np.array(
        [(item, feature) for item in items for feature in items[item]]
    )
    user_opinion_pairs = np.array(
        [(user, feature) for user in users for feature in users[user]]
    )
    return item_aspect_pairs, user_opinion_pairs, items.keys(), users.keys()


at_feedback = amazon_toy.load_feedback()
at_feedback = at_feedback[: len(at_feedback) // 20]  # reduce data size
at_sentiment = amazon_toy.load_sentiment()
items_feature, users_feature, items_list, users_list = (
    create_item_features_from_aspects(at_sentiment)
)
# remove unknown users and items from rating data
at_feedback_excl_unknowns = [
    x for x in at_feedback if x[0] in users_list and x[1] in items_list
]

# prepare data
rs = RatioSplit(
    data=at_feedback,
    test_size=0.2,
    item_feature=FeatureModality(items_feature),
    # user_feature=FeatureModality(users_feature), # user feature is not used in this experiment
    seed=42,
    exclude_unknowns=True,
)
# initialize recommenders, explainers and metrics
fm = FMRec()
limers = Exp_LIMERS(rec_model=fm, dataset=rs.train_set)
pspnfns = PSPN()
fdiv = DIV()
fpr = FPR()

# initialize experiment
models = [(fm, limers)]
metrics = [pspnfns, fdiv, fpr]
experiment = Experiment_Explainers(
    eval_method=rs,
    models=models,
    metrics=metrics,
    distribution=True,
    rec_k=4,
    feature_k=4,
    eval_train=True,
)
experiment.run()
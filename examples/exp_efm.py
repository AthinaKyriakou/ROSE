from cornac.experiment import Experiment_Explainers
from cornac.models import EFM
from cornac.explainer import Exp_EFM
from cornac.metrics_explainer import Metric_Exp_DIV
from cornac.datasets import amazon_toy
from cornac.data.reader import Reader
from cornac.eval_methods import RatioSplit
from cornac.data.sentiment import SentimentModality

rating = amazon_toy.load_feedback(fmt="UIRT", reader=Reader(min_user_freq=50))
sentiment_data = amazon_toy.load_sentiment(reader=Reader(min_user_freq=50))
rating = rating[:500]

md = SentimentModality(data=sentiment_data)

eval_method = RatioSplit(
    data=rating,
    group_by="user",
    chrono=True,
    sentiment=md,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
)

efm = EFM(max_iter=20)

efm_exp = Exp_EFM(rec_model=efm, dataset=eval_method.train_set)

# Get explanations for UI pair
# user = eval_method.train_set.user_ids[2]
# item = eval_method.train_set.item_ids[6]
# explanation = efm_exp.explain_one_recommendation_to_user(user, item)
# print(f"User: {user}, Item:{item} - Explanation: {explanation}")

# Evaluate explanations
div = Metric_Exp_DIV()
models = [(efm, efm_exp)]
metrics = [div]
# initialize experiment
experiment = Experiment_Explainers(
    eval_method=eval_method,
    models=models,
    metrics=metrics,
    rec_k=10,
    distribution=False,
    feature_k=10,
    eval_train=True,
)
experiment.run()
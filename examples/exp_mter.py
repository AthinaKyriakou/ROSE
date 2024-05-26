from cornac.experiment import Experiment_Explainers
from cornac.models import MTER
from cornac.explainer import Exp_MTER
from cornac.metrics_explainer import Metric_Exp_FPR
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

mter = MTER(max_iter=20,
            n_user_factors=8,
            n_item_factors=8,
            n_aspect_factors=8,
            n_opinion_factors=8)

mter.fit(eval_method.train_set)

mter_exp = Exp_MTER(rec_model=mter, dataset=eval_method.train_set)

# Get explanations for UI pair
user = eval_method.train_set.user_ids[2]
item = eval_method.train_set.item_ids[6]

explanation = mter_exp.explain_one_recommendation_to_user(user, item)
print(f"User: {user}, Item:{item} - Explanation: {explanation}")

# Evaluate explanations
fpr_with_input_as_groundtruth = Metric_Exp_FPR(ground_truth=sentiment_data)
models = [(mter, mter_exp)]
metrics = [fpr_with_input_as_groundtruth]
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
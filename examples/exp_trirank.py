from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit
from cornac.data.reader import Reader
from cornac.experiment import Experiment_Explainers
from cornac.models import TriRank
from cornac.explainer import Exp_TriRank
from cornac.metrics_explainer import Metric_Exp_DIV
from cornac.metrics import NDCG, AUC

# Load rating and sentiment information
rating = amazon_toy.load_feedback(reader=Reader(min_user_freq=50))
sentiment = amazon_toy.load_sentiment(reader=Reader(min_user_freq=50))


# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiment)


# Define an evaluation method to split feedback into train and test sets
eval_method = RatioSplit(
    data=rating,
    test_size=0.15,
    exclude_unknowns=True,
    verbose=True,
    sentiment=md,
    seed=123,
)

# Instantiate the model
trirank = TriRank(
    verbose=True,
    seed=123,
)
# Instantiate the explainer
exp_trirank = Exp_TriRank(rec_model=trirank, dataset=eval_method.train_set)

# Instantiate evaluation metrics
ndcg_50 = NDCG(k=50)
auc = AUC()

# initialize metrics for explainers
div = Metric_Exp_DIV()

# Run the experiment for the TriRank model with the explainer
Experiment_Explainers(
    eval_method=eval_method,
    models=[(trirank, exp_trirank)],
    metrics=[div],
    rec_k=10,
    feature_k=10,
    eval_train=True,
    distribution=False,
).run()
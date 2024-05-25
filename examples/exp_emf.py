from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data.reader import Reader
from cornac.models import EMF
from cornac.explainer import Exp_EMF

# Load MovieLens
data = movielens.load_feedback(variant="100K", reader=Reader(min_user_freq=150))

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=data, test_size=0.2, exclude_unknowns=False, verbose=True
)

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

emf.fit(ratio_split.train_set)
exp_emf = Exp_EMF(rec_model=emf, dataset=ratio_split.train_set)

user = ratio_split.train_set.user_ids[0]
item = ratio_split.train_set.item_ids[0]
exp_emf.explain_one_recommendation_to_user(user, item)


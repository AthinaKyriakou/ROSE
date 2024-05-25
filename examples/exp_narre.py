from cornac.datasets import amazon_digital_music
from cornac.eval_methods import RatioSplit
from cornac.data.reader import Reader
from cornac.data import ReviewModality
from cornac.data.text import BaseTokenizer
from cornac.experiment import Experiment_Explainers
from cornac.models import NARRE
from cornac.explainer import Exp_NARRE
from cornac.metrics_explainer import Metric_Exp_DIV


feedback = amazon_digital_music.load_feedback(reader=Reader(min_user_freq=50))
reviews = amazon_digital_music.load_review(reader=Reader(min_user_freq=50))


review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    review_text=review_modality,
    verbose=True,
    seed=123,
)

pretrained_word_embeddings = {}  # You can load pretrained word embedding here

narre = NARRE(
    embedding_size=100,
    id_embedding_size=32,
    n_factors=32,
    attention_size=16,
    kernel_sizes=[3],
    n_filters=64,
    dropout_rate=0.5,
    max_text_length=50,
    batch_size=64,
    max_iter=10,
    init_params={'pretrained_word_embeddings': pretrained_word_embeddings},
    verbose=True,
    seed=123,
)

narre_exp = Exp_NARRE(rec_model=narre, dataset=ratio_split.train_set)

div = Metric_Exp_DIV()

experiment = Experiment_Explainers(
    eval_method=ratio_split,
    models=[(narre, narre_exp)],
    metrics=[div],
    rec_k=10,
    feature_k=10,
    eval_train=True,
    distribution=False,
)

experiment.run()

"""
This data is built based on the GoodReads dataset
"""
from ..data import Reader
from typing import List

import numpy as np
import pandas as pd
from ..data import FeatureModality, SentimentModality
from ..eval_methods import RatioSplit


def load_feedback(fpath, fmt="UIR", sep=',', skip_lines=0, reader: Reader = None) -> List:
    """Load the user-item ratings, scale: [1,5]

    Parameters
    ----------
    fpath: file path to xx-rating.txt
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=sep, skip_lines=skip_lines)

def load_sentiment(fpath, reader: Reader = None) -> List:
    """Load the user-item-sentiments
    The dataset was constructed by the method described in the reference paper.

    Parameters
    ----------
    fpath: file path to xx-sentiment.txt
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, [(aspect, opinion, sentiment), (aspect, opinion, sentiment), ...]).

    References
    ----------
    Cornac.data.amazon_toy
    Gao, J., Wang, X., Wang, Y., & Xie, X. (2019). Explainable Recommendation Through Attentive Multi-View Learning. AAAI.
    """
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UITup', sep=',', tup_sep=':')


def prepare_data(data_name = "goodreads",test_size=0.2, dense=False, verbose=False, seed=42, item=True, user=False,sample_size=0.1):
    fpath_uir_dense = 'cornac/datasets/good_reads/good_read_dense.csv'
    sep_rating = ','
    skip_lines = 0
    if verbose:
        print('Preparing data...')
    if data_name == 'goodreads':
        fpath_sentiment = 'cornac/datasets/good_reads/goodreads_sentiment.txt'
        fpath_rating = 'cornac/datasets/good_reads/goodreads_rating.txt'

        if dense:
            fpath_rating = fpath_uir_dense
            sep_rating = '\t'
            skip_lines = 1
        sentiment = load_sentiment(fpath = fpath_sentiment)
        
        sentiment_modality = SentimentModality(data = sentiment)
        rating = load_feedback(fpath = fpath_rating, sep = sep_rating, skip_lines = skip_lines)
        indices = np.random.choice(len(rating), int(len(rating)*sample_size), replace=False)
        rating = np.array(rating)[indices]
        rs = RatioSplit(data=rating, test_size=test_size, exclude_unknowns=True, sentiment=sentiment_modality, verbose=verbose, seed=seed)

    elif data_name == 'goodreads_uir':
        fpath_uir = 'cornac/datasets/good_reads/good_read_UIR_sample.csv'
        df = pd.read_csv(fpath_uir, sep='\t', header=0, names=['user_id', 'item_id', 'rating'])
        df = df.sample(frac=sample_size)
        data = df[['user_id', 'item_id', 'rating']].values
        rs = RatioSplit(data=data, test_size=test_size, verbose=verbose, seed=seed)
        
    elif data_name == 'goodreads_uir_1000':
        fpath_uir = 'cornac/datasets/good_reads/good_read_UIR_1000.csv'
        if dense:
            fpath_uir = fpath_uir_dense
        df = pd.read_csv(fpath_uir, sep='\t', header=0, names=['user_id', 'item_id', 'rating'])
        df = df.sample(frac=sample_size)
        data = df[['user_id', 'item_id', 'rating']].values
        rs = RatioSplit(data=data, test_size=test_size, verbose=verbose, seed=seed)

    elif data_name == "goodreads_limers":
        fpath_uir = 'cornac/datasets/good_reads/good_read_UIR_sample.csv'
        #fpath_uir = 'cornac/datasets/good_reads/good_read_UIR_1000.csv'
        fpath_genres = 'cornac/datasets/good_reads/goodreads_genres.csv'
        fpath_aspects = 'cornac/datasets/good_reads/uid_aspect_features.txt'
        if dense:
            fpath_uir = fpath_uir_dense
        #df = pd.read_csv(fpath_uir, header=0, names=['user_id', 'item_id', 'rating'])
        df = pd.read_csv(fpath_uir, sep='\t', header=0, names=['user_id', 'item_id', 'rating'])
        if item==True:
            genres = pd.read_csv(fpath_genres)
            item_features = np.array([[x,y] for [x,y] in zip(genres['item_id'].to_numpy(), genres['feature'].to_numpy())])
            df = df[df['item_id'].isin(genres['item_id'])]
        if user==True:
            user_aspects = pd.read_csv(fpath_aspects, sep='\t', usecols=['user_id', 'feature'])
            user_features = np.array([[x,y] for [x,y] in zip(user_aspects['user_id'].to_numpy(), user_aspects['feature'].to_numpy())])
            df = df[df['user_id'].isin(user_aspects['user_id'])]
        df = df.sample(frac=sample_size)
        #df = pd.read_csv(fpath_rating, dtype={"user_id":str,"item_id":str})
        data_triple = df[['user_id', 'item_id', 'rating']].values
        if item==True and user==True:
            rs = RatioSplit(data=data_triple, seed=seed, item_feature = FeatureModality(item_features), user_feature = FeatureModality(user_features), test_size=test_size, exclude_unknowns=True)
        elif item==True:
            rs = RatioSplit(data=data_triple, seed=seed, item_feature = FeatureModality(item_features), test_size=test_size, exclude_unknowns=True)
        else:
            rs = RatioSplit(data=data_triple, seed=seed, user_feature = FeatureModality(user_features), test_size=test_size, exclude_unknowns=True)
    else:
        print(f'No dataset named {data_name}')
        return None
    if verbose:
        print('Data prepared.')
    return rs

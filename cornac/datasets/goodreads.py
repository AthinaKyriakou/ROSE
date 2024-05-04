
"""
This data is built based on the GoodReads dataset.  
"""
from ..utils import cache
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

def load_sentiment(reader: Reader = None) -> List:
    """Load the user-item-sentiments
    The dataset was constructed by the method described in the reference paper.

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, [(aspect, opinion, sentiment), (aspect, opinion, sentiment), ...]).

    References
    ----------
    [1] Gao, J., Wang, X., Wang, Y., & Xie, X. (2019). Explainable Recommendation Through Attentive Multi-View Learning. AAAI.
    """
    fpath_sentiment = cache(url='https://zenodo.org/records/11061007/files/goodreads_sentiment.txt?download=1')
    reader = Reader() if reader is None else reader
    return reader.read(fpath_sentiment, fmt='UITup', sep=',', tup_sep=':')


def prepare_data(data_name = "goodreads",test_size=0.2, dense=False, verbose=False, seed=42, item=True, user=False,sample_size=0.1):
    """Prepare data for the GoodReads dataset. 
    Generate the data split for the dataset.

    Parameters
    ----------
    data_name: str, default: 'goodreads'
        Name of the dataset to be prepared.
        
        Options: 'goodreads', 'goodreads_uir', 'goodreads_uir_1000', 'goodreads_limers'
        
        - 'goodreads': user-item-rating with sentiment data.
        
        - 'goodreads_uir': user-item-rating data in the whole dataset.
        
        - 'goodreads_uir_1000': user-item-rating data with 1000 samples.
        
        - 'goodreads_limers': user-item-rating data with item genres and user aspects.
        
    test_size: float, default: 0.2
        The proportion of the dataset to include in the test split.
    dense: bool, default: False
        If True, use the dense version of the dataset.
    verbose: bool, default: False
        If True, print out messages.
    seed: int, default: 42
        Random seed.
    item: bool, default: True
        If True, include item genres when preparing 'goodreads_limers'.
    user: bool, default: False
        If True, include user aspects when preparing 'goodreads_limers'.
    sample_size: float, default: 0.1
        The proportion of the dataset to include in the split.
        
    Returns
    -------
    rs: `obj:cornac.eval_methods.RatioSplit`
        The data split.
    """
    # fpath_uir_dense = 'cornac/datasets/good_reads/good_read_dense.csv'
    fpath_uir_dense = cache(url='https://zenodo.org/records/11061007/files/good_read_dense.csv?download=1')
    sep_rating = ','
    skip_lines = 0
    if verbose:
        print('Preparing data...')
    if data_name == 'goodreads':
        if dense:
            fpath_rating = fpath_uir_dense
            sep_rating = '\t'
            skip_lines = 1
        else:
            fpath_rating = cache(url='https://zenodo.org/records/11061007/files/goodreads_rating.txt?download=1')
        sentiment = load_sentiment()
        sentiment_modality = SentimentModality(data = sentiment)
        rating = load_feedback(fpath_rating, sep = sep_rating, skip_lines = skip_lines)
        indices = np.random.choice(len(rating), int(len(rating)*sample_size), replace=False)
        rating = np.array(rating)[indices]
        rs = RatioSplit(data=rating, test_size=test_size, exclude_unknowns=True, sentiment=sentiment_modality, verbose=verbose, seed=seed)

    elif data_name == 'goodreads_uir':
        fpath_uir = cache(url='https://zenodo.org/records/11061007/files/good_read_UIR_sample.csv?download=1')
        df = pd.read_csv(fpath_uir, sep='\t', header=0, names=['user_id', 'item_id', 'rating'])
        df = df.sample(frac=sample_size)
        data = df[['user_id', 'item_id', 'rating']].values
        rs = RatioSplit(data=data, test_size=test_size, verbose=verbose, seed=seed)
        
    elif data_name == 'goodreads_uir_1000':
        fpath_uir = cache(url='https://zenodo.org/records/11061007/files/good_read_UIR_1000.csv?download=1')
        if dense:
            fpath_uir = fpath_uir_dense
        df = pd.read_csv(fpath_uir, sep='\t', header=0, names=['user_id', 'item_id', 'rating'])
        df = df.sample(frac=sample_size)
        data = df[['user_id', 'item_id', 'rating']].values
        rs = RatioSplit(data=data, test_size=test_size, verbose=verbose, seed=seed)

    elif data_name == "goodreads_limers":
        fpath_uir = cache(url='https://zenodo.org/records/11061007/files/good_read_UIR_sample.csv?download=1')
        #fpath_uir = cache(url='https://zenodo.org/records/11061007/files/good_read_UIR_1000.csv?download=1')
        fpath_genres = cache(url='https://zenodo.org/records/11061007/files/goodreads_genres.csv?download=1')
        fpath_aspects = cache(url='https://zenodo.org/records/11061007/files/uid_aspect_features.txt?download=1')
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

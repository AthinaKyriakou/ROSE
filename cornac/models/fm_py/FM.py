import copy
import logging
import numpy as np
import pandas as pd
from pyfm import pylibfm
from cornac.models.recommender import Recommender
from cornac.data.dataset import Dataset
from scipy import sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _prepare_data(train_set):
    """
    Reformat cornac dataset with format (user, item, rating) tuples to 
        1. training_df of df (user, item);
        2. y_train of array of ratings;
        3. map cornac feature modality from (item_id, feature) to (item_idx, feature) using iid_map
    Args: cornac dataset (rs.train_set, rs.test_set or rs.val_set)
            user_id and item_id are treated as with dtype = string
    """
    training_df = pd.DataFrame({'user_id':list(map(str,train_set.uir_tuple[0])), 'item_id': list(map(str, train_set.uir_tuple[1]))})
    y_train = train_set.uir_tuple[2]
    item_info_short = pd.DataFrame()
    user_info_short = pd.DataFrame()
    if train_set.item_feature: 
        item_info_short = pd.DataFrame({'item_id':list(map(str,train_set.item_feature.features[:,0])), 'feature': list(map(str, train_set.item_feature.features[:,1]))})
        ##rs.item_features keep already item_id 
        ##update item_id to item_indices so training_df and item_feature can be merged into one dataframe
        iid_map = pd.DataFrame({'item_id': list(map(str, train_set.iid_map.keys())), 'item_indices': list(map(str, train_set.iid_map.values()))})
        item_info_short = item_info_short.merge(iid_map, on='item_id', how='left')
        item_info_short = item_info_short.dropna().drop(columns=['item_id'])
        item_info_short.columns = ['feature', 'item_id']
        if len(training_df) > len(training_df[training_df['item_id'].isin(item_info_short['item_id'])]):
            raise ValueError(
                "training data contain items which features are unknown"
            )
    if train_set.user_feature:
        user_info_short = pd.DataFrame({'user_id':list(map(str,train_set.user_feature.features[:,0])), 'feature': list(map(str, train_set.user_feature.features[:,1]))})
        ##rs.item_features keep already item_id 
        ##update item_id to item_indices so training_df and item_feature can be merged into one dataframe
        uid_map = pd.DataFrame({'user_id': list(map(str, train_set.uid_map.keys())), 'user_indices': list(map(str, train_set.uid_map.values()))})
        user_info_short = user_info_short.merge(uid_map, on='user_id', how='left')
        user_info_short = user_info_short.dropna().drop(columns=['user_id'])
        user_info_short.columns = ['feature', 'user_id']
        if len(training_df) > len(training_df[training_df['user_id'].isin(user_info_short['user_id'])]):
            raise ValueError(
                "training data contain users which features are unknown"
            )

    return training_df, y_train, item_info_short, user_info_short
  
class FMRec(Recommender):
    """Factoriazation machine recommender algorithm
    Parameters
    ------------
        name: str
            recommender name
        trainable: bool
            whether model can be trained
        verbose: bool
            Whether or not to print current iteration, training error
        users_features: bool
            whether features are used
        num_factors: int
            The dimensionality of the factorized 2-way interactions
        num_iter: int
            Number of iterations
        k0: bool
            Use bias. Defaults to true.
        k1: bool
            Use 1-way interactions (learn feature weights).
            Defaults to true.
        init_stdev: double, optional
            Standard deviation for initialization of 2-way factors.
            Defaults to 0.01.
        validation_size: double, optional
            Proportion of the training set to use for validation.
            Defaults to 0.01.
        power_t: double
            The exponent for inverse scaling learning rate [default 0.5].
        t0: double
            Constant in the denominator for optimal learning rate schedule.
            Defaults to 0.001.
        task: str
            regression: Labels are real values.
            classification: Labels are either positive or negative.
        initial_learning_rate: double
            Defaults to 0.01
        learning_rate_schedule: str
            The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)
    """
    def __init__(self, name='fm_regressor', trainable=True, verbose=True, uses_features=True, 
                 num_factors=50, num_iter=10, k0=True, k1=True, init_stdev=0.1, validation_size = 0.01, power_t = 0.5, t0=0.001, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal"):
        
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.uses_features = uses_features
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.k0 = k0
        self.k1 = k1
        self.init_stdev = init_stdev
        self.validation_size = validation_size
        self.power_t = power_t
        self.t0 = t0
        self.task = task
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.one_hot_columns = None

        # default rec
        self.fm = pylibfm.FM(
            num_factors=self.num_factors,
            num_iter=self.num_iter,
            k0=self.k0,
            k1=self.k1,
            init_stdev = self.init_stdev,
            validation_size=self.validation_size,
            power_t=self.power_t,
            t0=self.t0,
            task=self.task,
            initial_learning_rate=self.initial_learning_rate,
            learning_rate_schedule=self.learning_rate_schedule,
            verbose=self.verbose
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        #TODO: modify item_features to wide format. OR add a function in Dataset to perform this transformation??

        self.train_set = self.LimeRSDataset(train_set)
        self.num_users = self.train_set.num_users
        self.num_items = self.train_set.num_items
        self.uid_map = self.train_set.uid_map
        self.iid_map = self.train_set.iid_map
        self.min_rating = self.train_set.min_rating
        self.max_rating = self.train_set.max_rating
        self.global_mean = self.train_set.global_mean
        df = self.train_set.merge_uir_with_features(self.train_set.training_df, self.uses_features)
        training_data, training_columns = self.train_set.convert_to_pyfm_format(df)  
        self.one_hot_columns = training_columns

        self.fm.fit(training_data, self.train_set.y_train) #training_data is a sparse matrix; y_train is an array
        self.val_set = None if val_set is None else self.LimeRSDataset(val_set)
        self.is_fitted = True
        return self

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.
            
        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_id is not None:
            df = pd.DataFrame({"user_id": str(user_id), "item_id": str(item_id)},index=[0])
        else:
            all_items = [str(item_idx) for _, item_idx in self.train_set.iid_map.items()]
            df = pd.DataFrame({'user_id':[str(user_id) for _ in all_items], "item_id": all_items})

        df = self.train_set.merge_uir_with_features(df, self.uses_features)

        all_predictions = list()

        # divide in chunks to avoid memory errors
        chunk_size = 10
        chunks = np.array_split(df, chunk_size) if len(df) > 10 else [df]
        for chunk in chunks:
            # convert
            test_data, _ = self.train_set.convert_to_pyfm_format(chunk)

            # get predictions
            preds = self.fm.predict(test_data)
            all_predictions.extend(preds.round(3))

        return all_predictions[0] if item_id != None else np.array(all_predictions)
    
    def score_neighbourhood(self, neighborhood_df):
        """make prediction on a list of items for each user. dataframe of [user_idx, item_idx] are
        passed and predictions are made by chunks. 
        This is to void iterating each row and speed up the explain_instance function in limers 

        Returns: numpy array of predictions ordred by item_idx
        """
        
        df = self.train_set.merge_uir_with_features(neighborhood_df, self.uses_features)
        all_predictions = list()

        # divide in chunks to avoid memory errors
        chunk_size = 10
        chunks = np.array_split(df, chunk_size)
        for chunck in chunks:
            # convert
            test_data, _ = self.train_set.convert_to_pyfm_format(chunck)

            # get predictions
            preds = self.fm.predict(test_data)
            all_predictions.extend(preds.round(3))

        return np.array(all_predictions)
    
    class LimeRSDataset(Dataset):
        """Dataset object used for FM and Limers model training"""
        def __init__(self, dataset):
            """
            reformat rs dataset so to be able to merge uir with features

            """
            uir_tuples = dataset.uir_tuple
            Dataset.__init__(self, num_users=dataset.num_users, num_items=dataset.num_items,
                    uid_map=dataset.uid_map, iid_map=dataset.iid_map, uir_tuple=uir_tuples)
            ### TODO: check error when running fm_unittest -- "AttributeError: 'Dataset' object has no attribute 'total_users'" --yingying
            
            training_df, y_train, train_item_features, train_user_features = _prepare_data(dataset)
            self.training_df = training_df
            self.y_train = y_train
            self.user_frequency = self.set_train_frequency(item='False')
            self.item_frequency = self.set_train_frequency(item='True')
            self.item_features = train_item_features
            self.user_features = train_user_features
        
        @staticmethod
        def convert_to_pyfm_format(df, columns=None):
            """convert dataframe to sparse matrix format
            Return: sparse matrix, one hot encoded column names
            """
            df_ohe = pd.get_dummies(df)
            if columns is not None:
                df_ohe = df_ohe.reindex(columns=columns)
                df_ohe = df_ohe.fillna(0)
            data_sparse = sparse.csr_matrix(df_ohe.astype(np.float64))  
            data_sparse = data_sparse.astype(np.float64)

            return data_sparse, df_ohe.columns
        
        def convert_to_feature_long(self):
            """
            Convert features from row to column representation
            example: [(feature 1, item_idx), (feature 2, item_idx)] -> [item_idx, feature1, feature2] 
            """
            item_features_wide = pd.DataFrame()
            user_features_wide = pd.DataFrame()
            if not self.item_features.empty:
                # if dataset contains item_feature
                item_features_wide = self.item_features.copy()
                item_features_wide['value'] = 1
                item_features_wide = item_features_wide.pivot(index='item_id', columns= 'feature', values='value').reset_index().fillna(0)
            if not self.user_features.empty:
                # if dataset contains user_feature
                user_features_wide = self.user_features.copy()
                user_features_wide['value'] = 1
                user_features_wide = user_features_wide.pivot(index='user_id', columns= 'feature', values='value').reset_index().fillna(0)
            return item_features_wide, user_features_wide
        
        def merge_uir_with_features(self, train_df, uses_features=True):
            if uses_features:
                #item_features are set as vectors, add a value column and convert to long format
                if self.user_features.empty:
                    item_features, _ = self.convert_to_feature_long()
                    df = pd.merge(
                        train_df,
                        item_features,
                        on="item_id",
                        how="left",
                )
                elif self.item_features.empty:
                    _, user_features = self.convert_to_feature_long()  
                    df = pd.merge(
                        train_df,
                        user_features,
                        on="user_id",
                        how="left",
                )          
                else:
                    item_features, user_features = self.convert_to_feature_long()
                    df = pd.merge(pd.merge(train_df,
                                        item_features,
                                        on="item_id",
                                        how="left"),
                                    user_features,
                                    on='user_id',
                                    how='left',
                                    suffixes=('_i_f', '_u_f')
                                    )
            else:
                df = (train_df.copy()) 
            return df

        def set_train_frequency(self, item='True'):
            """calculate user or item frequency appeared in the training set, frequency is mapped against id"""
            if item == 'True':
                frequency = self.training_df['item_id'].value_counts().to_dict()
                frequency = [(key, frequency[str(value)]) for key, value in self.iid_map.items() if str(value) in frequency]
                frequency = sorted(frequency, key=lambda x: x[1], reverse=True)
            else:
                frequency = self.training_df['user_id'].value_counts().to_dict()
                frequency = [(key, frequency[str(value)]) for key, value in self.uid_map.items() if str(value) in frequency]
                frequency = sorted(frequency, key=lambda x: x[1], reverse=True)
            return frequency
    
        
        def pick_top_users(self, count=1, train='True'):
            """pick the top n users based on train/test user frequency"""
            if train:
                return [x for x,_ in self.user_frequency[:count]]
            else:
                pass

        def pick_top_items(self, count=1,train='True'):
            """pick the top n items based on train/test item frequency"""
            if train: 
                return [x for x,_ in self.item_frequency[:count]]
            else:
                pass
        
        def map_to_df(self, item="True"):
            """helper function to map item_idx to item_id or user_idx to user_id; returned df has xid.map.keys() as id and xid.map.items() as idx"""
            if item == "True":
                df = pd.DataFrame({'item_id': list(self.iid_map.keys()), 'item_idx': list(self.iid_map.values())})
                df['item_idx'] = df['item_idx'].astype(str)
            else:
                df = pd.DataFrame({'user_id': list(self.uid_map.keys()), 'user_idx': list(self.uid_map.values())})
                df['user_idx'] = df['user_idx'].astype(str)
            return df
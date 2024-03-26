import pandas as pd
import numpy as np


class LibffmModConverter(object):
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.col_rating = None
        self.field_names = None
        self.field_count = None
        self.feature_count = None

    def fit(self, df, col_rating='rating'):
        """Fit the dataframe for libffm format.
        This method does nothing but check the validity of the input columns

        Args:
            df (pandas.DataFrame): input Pandas dataframe.
            col_rating (str): rating of the data.

        Return:
            object: the instance of the converter
        """

        # Check column types.
        types = df.dtypes
        if not all(
            [
                x == object or np.issubdtype(x, np.integer) or x == np.float
                for x in types
            ]
        ):
            raise TypeError("Input columns should be only object and/or numeric types.")

        if col_rating not in df.columns:
            raise TypeError(
                "Column of {} is not in input dataframe columns".format(col_rating)
            )

        self.col_rating = col_rating
        self.field_names = list(df.drop(col_rating, axis=1).columns)

        return self
    
    def transform(self, df):
        """Tranform an input dataset with the same schema (column names and dtypes) to libffm format
        by using the fitted converter.

        Args:
            df (pandas.DataFrame): input Pandas dataframe.

        Return:
            pandas.DataFrame: Output libffm format dataframe.
        """
        
        if self.col_rating not in df.columns:
            raise ValueError(
                "Input dataset does not contain the label column {} in the fitting dataset".format(
                    self.col_rating
                )
            )

        if not all([x in df.columns for x in self.field_names]):
            raise ValueError(
                "Not all columns in the input dataset appear in the fitting dataset"
            )

        # Encode field-feature.
        idx = 1
        self.field_feature_dict = {}
        for field in self.field_names:
            for feature in df[field].values:
                # Check whether (field, feature) tuple exists in the dict or not.
                # If not, put them into the key-values of the dict and count the index.
                if (field, feature) not in self.field_feature_dict:
                    self.field_feature_dict[(field, feature)] = idx
                    if df[field].dtype == object:
                        idx += 1
            if df[field].dtype != object:
                idx += 1

        self.field_count = len(self.field_names)
        self.feature_count = idx - 1

        df = self.tabulate(df)

        field_idx = {col:idx+1 for idx, col in enumerate(self.field_names)}

        def _convert_features(field_feature,x):
            return "{}:{}:{}".format(field_idx[field_feature[0]], self.field_feature_dict[field_feature],x)

        def _convert_key(field, feature, field_index):
            field_feature_index = self.field_feature_dict[(field, feature)]
            value = 1
            return "{}:{}:{}".format(field_index, field_feature_index,value)

        for col in df.columns:
            df[col] = df[col].apply(lambda x: _convert_features(col, x))

        df = df.reset_index()
        for col_index, col in enumerate(['user_id', 'item_id']):
            df[col] = df[col].apply(lambda x: _convert_key(col, x, col_index+1))

        # # Move rating column to the first.
        column_names = list(df.drop(self.col_rating, axis=1).columns)
        column_names.insert(0, self.col_rating)
        df = df[column_names]

        if self.filepath is not None:
            np.savetxt(self.filepath, df.values, delimiter=" ", fmt="%s")

        return df  
    
    @staticmethod
    def tabulate(df):
        """
        to transform the dataframe from long format to wide format with two level headers: field and features
        """
        df = df.melt(id_vars = ['user_id', 'item_id', 'rating'], var_name='field', value_name='features').drop_duplicates()
        df['value'] = "1"
        df = df.pivot(index=['user_id', 'item_id', 'rating'], columns=['field', 'features'], values='value').fillna("0")
        df.columns = df.columns.to_flat_index()
        return df


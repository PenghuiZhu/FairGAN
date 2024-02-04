import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Dataset:
    def __init__(self, dataframe, target_names, protected_names):
        """
        Initializes dataset class given a Pandas DataFrame.

        Parameters:
        - dataframe (Pandas DataFrame): Training data before pre-processing.
        - target_names (List[str]): Target variable column names.
        - protected_names (List[str]): Protected variable column names.

        Attributes:
        - scaler (sklearn.preprocessing.MinMaxScaler): MinMaxScaler object used to scale data.
        - np_data (numpy.ndarray): Processed data in a Numpy array format.
        - processed_col_types (List[str]): List of the original dataset column types, used in post-processing.
        """
        self.dataframe = dataframe
        self.target_names = target_names
        self.protected_names = protected_names
        self.scaler = MinMaxScaler()
        self.np_data = None
        self.processed_col_types = []



    def pre_process(self, protected_var, outcome_var, output_file_name_path,
                    multiclass=False, min_max_scale=True):
        """
        Basic pre-processing on a Pandas DataFrame including one-hot encoding,
        scaling, checking for nulls, etc. Saves a pickle file with a numpy array
        and a csv file with a data dictionary in the specified path.

        Parameters:
        - protected_var (str): Name of the protected column in the Pandas DataFrame.
        - outcome_var (str): Name of the outcome column in the Pandas DataFrame.
        - output_file_name_path (str): Path to save the Pickle and CSV files.
        - multiclass (bool, optional): Set to True if the protected variable is categorical and has more than two states. Defaults to False.
        - min_max_scale (bool, optional): Set to False if using scaled data. Defaults to True.

        Raises:
        - Exception: If dataset has nulls.

        Returns:
        - np_data (numpy.ndarray): Numpy array with pre-processed data.
        """
        if self.dataframe.isnull().values.any():
            raise Exception("Dataset contains null values.")

        # Drop outcome variable from features
        X = self.dataframe.drop([outcome_var], axis=1)
        y = self.dataframe[outcome_var]

        # One-hot encode categorical variables if multiclass is True
        if multiclass:
            X = pd.get_dummies(X, columns=[protected_var], drop_first=True)
        else:
            # Ensure the protected variable is binary encoded
            X[protected_var] = X[protected_var].astype('category').cat.codes

        # Scale data if required
        if min_max_scale:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Combine features and target variable for saving
        processed_data = pd.concat([X, y], axis=1)
        np_data = processed_data.values

        # Save numpy array to pickle
        pickle_file_path = os.path.join(output_file_name_path, 'processed_data.pkl')
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(np_data, f)

        # Save data dictionary to CSV
        data_dict = {'Column': processed_data.columns.tolist(),
                     'Type': processed_data.dtypes.tolist()}
        data_dict_df = pd.DataFrame(data_dict)
        data_dict_csv_path = os.path.join(output_file_name_path, 'data_dictionary.csv')
        data_dict_df.to_csv(data_dict_csv_path, index=False)

        return np_data

    def post_process(self, gen_data_np):
        """
        Inverse scaling on the generated data from the trained model.

        Parameters:
        - gen_data_np (np.ndarray): Numpy array with generated data.

        Return Type:
        - gen_data_np (np.ndarray): Numpy array with post-processed data.
        """
        # Assume numerical features were scaled, and scaler is stored in self.scaler
        # Identify columns indexes that were scaled
        numerical_cols_indexes = [i for i, col_type in enumerate(self.processed_col_types) if
                                  col_type in ['int64', 'float64']]

        # Inverse transform the numerical columns
        gen_data_np[:, numerical_cols_indexes] = self.scaler.inverse_transform(
            gen_data_np[:, numerical_cols_indexes])

        return gen_data_np

    def get_protected_distribution(self, np_data):
        """
        Calculates the protected variable distribution after pre-processing.

        Parameters:
        - np_data (np.ndarray): Data in a numpy array.

        Return Type:
        - protected_distribution (List[float]): Distribution of each protected class.
        """
        # Assuming the protected variable was one-hot encoded and is located at a specific index
        # If protected variable was label encoded and located at index `protected_index`
        protected_index = self.dataframe.columns.get_loc(
            "sex")  # Adjust "protected_var_name_encoded" to actual encoded protected variable column name
        unique, counts = np.unique(np_data[:, protected_index], return_counts=True)
        protected_distribution = counts / np.sum(counts)

        return protected_distribution.tolist()

    def get_target_distribution(self, np_data):
        """
        Returns the target variable distribution after pre-processing.

        Parameters:
        - np_data (np.ndarray): Data in a numpy array.

        Return Type:
        - target_distribution (List[float]): Distribution of each target class.
        """
        # Assuming the target variable is located at the last column of np_data
        # Adjust this index if your target variable is located elsewhere
        target_index = -1  # Common if the target variable is the last column in your dataset

        # Extract the target column from np_data
        target_data = np_data[:, target_index]

        # Calculate the unique classes and their counts in the target variable
        unique, counts = np.unique(target_data, return_counts=True)

        # Calculate the distribution as the proportion of each class
        target_distribution = counts / np.sum(counts)

        return target_distribution.tolist()


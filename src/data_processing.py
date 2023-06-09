"""
data_processing.py
==================

This module contains the DataProcessing class, which provides methods to process
a golf dataset for a Naive Bayes classifier.

The DataProcessing class includes methods to:
- Create a fictional dataset for golf play decisions based on weather conditions.
- Encode categorical features into numerical ones.
- Add Laplace correction records to the dataset to avoid zero probabilities in
  the Naive Bayes calculations.

Classes:
--------
DataProcessing

"""

import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataProcessing:
    """
    A class to process the golf prediction dataset.
    """

    def __init__(self, df=pd.DataFrame()):
        """
        Initializes the DataProcessing class.

        Args:
            df (pd.DataFrame, optional): A dataframe containing the golf dataset. Defaults to None.
        """

        self.df = df

    @staticmethod
    def create_dataset():
        """
        Creates the fictional golf dataset using the given days, outlook, and play golf data.

        Returns:
            pd.DataFrame: The generated golf dataset as a DataFrame.
        """

        days_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        outlook_test = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny',
                        'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny']
        play_golf = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                     'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

        # create an NumPy array of the data
        combined_golf_data = np.column_stack(
            (days_test[:14], outlook_test, play_golf))

        # dataframe column Headers
        column_values = ['Day', 'Outlook', 'Play Golf']

        # Create a dataframe
        df = pd.DataFrame(data=combined_golf_data, columns=column_values)
        return df

    def encode_features(self):
        """
        Encodes the 'Outlook' and 'Play Golf' features using label encoding.

        Returns:
            tuple: A tuple containing the encoded 'Outlook' and 'Play Golf' features.
        """

        # create label encoders
        self.le_outlook = preprocessing.LabelEncoder()
        self.le_play = preprocessing.LabelEncoder()

        # Convert string labels into numbers
        outlook_encoded = self.le_outlook.fit_transform(self.df['Outlook'])
        play_encoded = self.le_play.fit_transform(self.df['Play Golf'])
        return outlook_encoded, play_encoded

    @staticmethod
    def add_laplace_correction(df):
        """
        Adds Laplace correction records to the golf dataset.

        Args:
            df (pd.DataFrame): The golf dataset.

        Returns:
            pd.DataFrame: The golf dataset with Laplace correction records added.
        """

        days_test = list(range(15, 21))
        outlook_laplace = ['Rainy', 'Rainy',
                           'Overcast', 'Overcast', 'Sunny', 'Sunny']
        play_laplace = ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']

        laplace_data = np.column_stack(
            (days_test, outlook_laplace, play_laplace))
        laplace_df = pd.DataFrame(data=laplace_data, columns=df.columns)
        df = pd.concat([df, laplace_df], ignore_index=True)

        return df

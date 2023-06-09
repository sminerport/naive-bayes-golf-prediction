"""
This module defines the NaiveBayesGolfModel class for the golf prediction project.

The NaiveBayesGolfModel class encapsulates a Naive Bayes classifier model from
the sklearn library and provides methods for training the model, making
predictions, and creating statistical tables related to the classification task.

The golf prediction project aims to predict the likelihood of playing golf
based on the weather outlook. It uses a Naive Bayes approach to compute
posterior probabilities based on prior knowledge and observed evidence. The
classification task is specifically focused on categorical data, making the
use of a Categorical Naive Bayes classifier suitable.

Classes:
--------
NaiveBayesGolfModel
    This class manages the training and prediction processes using a Naive
    Bayes classifier, and it creates frequency, likelihood, and posterior
    probability tables for data analysis.
"""

from sklearn.naive_bayes import CategoricalNB
import pandas as pd


class NaiveBayesGolfModel:
    """
    The NaiveBayesGolfModel class encapsulates a Naive Bayes classifier model
    that can be used to predict the likelihood of playing golf based on
    weather outlook.

    This class provides methods for:
    - Training a Naive Bayes model using a golf dataset.
    - Making predictions based on the trained model.
    - Creating and displaying various statistical tables such as the frequency
      table, likelihood table, and posterior probability table.

    The classifier used in this class is a Categorical Naive Bayes classifier
    from the sklearn library, which is suitable for features that are
    categorically distributed.

    Attributes:
    -----------
    model : CategoricalNB
        The Naive Bayes classifier model.
    outlook_play_df : pd.DataFrame
        The frequency table showing the distribution of 'Outlook' and 'Play Golf' features.
    outlook_likelihood_df : pd.DataFrame
        The likelihood table showing the probability of playing golf given the weather outlook.
    posterior_probability : pd.DataFrame
        The posterior probability table showing the revised probability of
        playing golf after considering the weather outlook.
    """

    def __init__(self, model=None):
        """
        Initializes the NaiveBayesGolfModel class.

        Args:
            model (CategoricalNB, optional): A Naive Bayes classifier model. Defaults to None.
        """
        self.model = model
        self.outlook_play_df = None
        self.outlook_likelihood_df = None
        self.posterior_probability = None

    def print_tables(self, df):
        """
        Prints the frequency, likelihood, and posterior probability tables for the given golf dataset.
        """

        # Calculate and print frequency, likelihood, and posterior tables
        self.outlook_play_df = self.create_frequency_table(df)
        self.outlook_likelihood_df = self.create_likelihood_table(
            self.outlook_play_df)
        self.posterior_probability = self.create_posterior_table(
            self.outlook_likelihood_df)

        print("\nFrequency table showing outlook x play:")
        print("---------------------------------------")
        print(self.outlook_play_df)

        print("\nLikelihood table for outlook: P(e | h) = P(Sunny | Yes):")
        print("--------------------------------------------------------")
        print(self.outlook_likelihood_df)

        print("\nPosterior probability: P(h | e) = P(Yes | Sunny):")
        print("-------------------------------------------------")
        print(self.posterior_probability)

    def train_model(self, outlook_encoded, play_encoded):
        """
        Trains the Naive Bayes classifier model using the encoded 'Outlook' and 'Play Golf' features.

        Args:
            outlook_encoded (numpy.ndarray): The encoded 'Outlook' feature.
            play_encoded (numpy.ndarray): The encoded 'Play Golf' feature.

        Returns:
            CategoricalNB: The trained Naive Bayes classifier model.
        """
        # create a categorical classifier
        # set alpha to 0 because we have already added laplace correction records
        self.model = CategoricalNB(alpha=1.0e-10)

        # reshape data for training
        outlook_encoded = outlook_encoded.reshape(-1, 1)

        # train the model using the training sets
        self.model.fit(outlook_encoded, play_encoded)
        return self.model

    def make_predictions(self, le_outlook, le_play):
        """
        Makes predictions based on weather outlook using the trained Naive Bayes classifier model.

        Args:
            le_outlook (LabelEncoder): A LabelEncoder object for the 'Outlook' feature.
            le_play (LabelEncoder): A LabelEncoder object for the 'Play Golf' feature.

        Returns:
            list: A list of tuples containing the condition, prediction, and probabilities.
        """
        predictions = []

        # for all categories of the feature variable, create predictions
        for condition in le_outlook.classes_:
            condition_transformed = le_outlook.transform([condition])
            yhat_prob = self.model.predict_proba([condition_transformed])
            prediction_statement = self.model.predict([condition_transformed])
            prediction = le_play.inverse_transform(prediction_statement)
            predictions.append((condition, prediction, yhat_prob))

        return predictions

    def create_frequency_table(self, df):
        """
        Creates a frequency table showing outlook x play.

        Args:
            df (pd.DataFrame): The golf dataset.

        Returns:
            pd.DataFrame: The frequency table.
        """

        outlook_play_df = pd.crosstab(
            index=df['Outlook'], columns=df['Play Golf'], margins=True)

        outlook_play_df.columns = list(
            outlook_play_df.iloc[:, :-1].columns) + ['Row Total']
        outlook_play_df.index = list(
            outlook_play_df.iloc[:-1, :].index) + ['Column Total']

        outlook_play_df = outlook_play_df.reindex(
            ['Sunny', 'Overcast', 'Rainy', 'Column Total'])
        outlook_play_df = outlook_play_df[['No', 'Yes', 'Row Total']]

        return outlook_play_df

    def create_likelihood_table(self, outlook_play_df):
        """
        Creates a likelihood table for outlook: P(e | h) = P(Sunny | Yes).

        Args:
            outlook_play_df (pd.DataFrame): The frequency table.

        Returns:
            pd.DataFrame: The likelihood table.
        """

        top_part = outlook_play_df.iloc[:-1, :] / \
            outlook_play_df.loc['Column Total']
        bottom_part = outlook_play_df.iloc[-1:, :] / \
            outlook_play_df.loc['Column Total', 'Row Total']
        outlook_likelihood_df = pd.concat([top_part, bottom_part])

        return outlook_likelihood_df

    def create_posterior_table(self, outlook_likelihood_df):
        """
        Creates a posterior probability table: P(h | e) = P(Yes | Sunny).

        Args:
            outlook_likelihood_df (pd.DataFrame): The likelihood table.

        Returns:
            pd.DataFrame: The posterior probability table.
        """

        posterior_probability = (
            outlook_likelihood_df.iloc[:-1, :] *
            outlook_likelihood_df.loc['Column Total']
        ).div(outlook_likelihood_df['Row Total'], axis=0)
        posterior_probability = posterior_probability.reindex(
            ['Sunny', 'Overcast', 'Rainy'])

        return posterior_probability

# model.py
from sklearn.naive_bayes import CategoricalNB

class NaiveBayesGolfModel:
    """
    A class to train and make predictions using a Naive Bayes classifier for the golf dataset.
    """

    def __init__(self, model=None):
        """
        Initializes the NaiveBayesGolfModel class.

        Args:
            model (CategoricalNB, optional): A Naive Bayes classifier model. Defaults to None.
        """
        self.model = model

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
        self.model = CategoricalNB(alpha=0)

        # reshape data for training
        outlook_encoded = outlook_encoded.reshape(-1,1)

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

"""
main.py
=======

This module contains the main function that orchestrates the process of
training and testing a Naive Bayes model for predicting whether golf will
be played based on weather conditions.

The main function follows these steps:
- Creates a fictional golf dataset.
- Applies Laplace correction to the dataset.
- Encodes categorical features into numerical ones.
- Trains a Naive Bayes model.
- Creates and displays a frequency table.
- Creates and displays a likelihood table.
- Creates and displays a posterior probability table.
- Makes predictions using the trained model.
- Prints the predictions, including the condition, the prediction itself,
  and the associated probabilities.

This script is meant to be run as the entry point of the program.

Functions:
----------
main
"""

import data_processing as dp
from model import NaiveBayesGolfModel



def main():
    # Create the dataset
    golf_df = dp.DataProcessing.create_dataset()
    print("Original dataset:")
    print("------------------")
    print(golf_df.head(14))
    print()

    # Add Laplace correction records
    golf_df = dp.DataProcessing.add_laplace_correction(golf_df)
    print("Dataset w/ Laplacian transformation:")
    print("------------------------------------")
    print(golf_df)

    # Encode the features
    dp_instance = dp.DataProcessing(golf_df)
    outlook_encoded, play_encoded = dp_instance.encode_features()

    # Train the model
    golf_model = NaiveBayesGolfModel()
    golf_model.train_model(outlook_encoded, play_encoded)

    freq_table = golf_model.create_frequency_table(golf_df)

    print("\nFrequency table showing outlook x play:")
    print("---------------------------------------")
    print(freq_table)

    print("\nLikelihood table for outlook: P(e | h) = P(Sunny | Yes):")
    print("--------------------------------------------------------")
    likelihood_table = golf_model.create_likelihood_table(freq_table)
    print(likelihood_table)

    print("\nPosterior probability: P(h | e) = P(Yes | Sunny):")
    print("-------------------------------------------------")
    posterior_table = golf_model.create_posterior_table(likelihood_table)
    print(posterior_table)

    # Make predictions
    predictions = golf_model.make_predictions(
        dp_instance.le_outlook, dp_instance.le_play)

    # Print predictions
    print("\nPredictions based on weather outlook:")
    print("-------------------------------------")
    for condition, prediction, yhat_prob in predictions:
        print(f"Condition: {condition}")
        print(f"Prediction (i.e., will we play golf?): {prediction}")
        print(f"Probabilities: {yhat_prob}")
        print()


if __name__ == "__main__":
    main()

# main.py
import data_processing as dp
from model import NaiveBayesGolfModel

def main():
    # Create the dataset
    golf_df = dp.DataProcessing.create_dataset()

    # Encode the features
    outlook_encoded, play_encoded, le_outlook, le_play = dp.encode_features(golf_df)

    # Train the model
    golf_model = NaiveBayesGolfModel()
    golf_model.train_model(outlook_encoded, play_encoded)

    # Make predictions
    predictions = golf_model.make_predictions(le_outlook, le_play)

    # Print predictions
    print("Predictions based on weather outlook:")
    print("-------------------------------------")
    for condition, prediction, yhat_prob in predictions:
        print(f"Condition: {condition}")
        print(f"Prediction (i.e., will we play golf?): {prediction}")
        print(f"Probabilities: {yhat_prob}")
        print()

if __name__ == "__main__":
    main()

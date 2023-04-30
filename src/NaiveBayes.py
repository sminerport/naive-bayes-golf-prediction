import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split

# create fictional dataset
days_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
outlook_test = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny']
play_golf = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# create an NumPy array of the data
combined_golf_data = np.column_stack((days_test[:14],outlook_test,play_golf))

# dataframe column Headers
column_values = ['Day', 'Outlook', 'Play Golf']

# Create a dataframe
df = pd.DataFrame(data = combined_golf_data, columns = column_values)

# Print the data with the laplace transformation recordss added
print("Original data set:")
print("------------------")
print(df.to_string(index=False))
print()
# Laplace correction records
outlook_laplace = ['Rainy', 'Rainy', 'Overcast', 'Overcast', 'Sunny', 'Sunny']
play_laplace = ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']

# Combine original data with laplace correction
outlook_test = outlook_test + outlook_laplace
play_golf = play_golf + play_laplace

# create an NumPy array of the data
combined_golf_data_laplace = np.column_stack((days_test,outlook_test,play_golf))

# Create a dataframe
df = pd.DataFrame(data = combined_golf_data_laplace, columns = column_values)

# Print the data with the laplace transformation recordss added
print("Dataset w/ Laplacian transformation:")
print("------------------------------------")
print(df.to_string(index=False))

# create frequency tables
outlook_play_df = pd.crosstab(index=df['Outlook'], columns=df['Play Golf'], margins = True)

# add row and column total headers
outlook_play_df.columns = list(outlook_play_df.iloc[:, :-1].columns) + ['Row Total']
outlook_play_df.index = list(outlook_play_df.iloc[:-1, :].index) + ['Column Total']

# rearrange the columns and rows
outlook_play_df = outlook_play_df.reindex(['Sunny', 'Overcast', 'Rainy', 'Column Total'])
outlook_play_df = outlook_play_df[['No', 'Yes', 'Row Total']]
print()

# Print the frequency table
print("Frequency table showing outlook x play:")
print("---------------------------------------")
print(outlook_play_df)
print()

# create and print the likelihood table
top_part = outlook_play_df.iloc[:-1, :]/outlook_play_df.loc['Column Total']
bottom_part = outlook_play_df.iloc[-1:, :]/outlook_play_df.loc['Column Total', 'Row Total']
outlook_likelihood_df = pd.concat([top_part, bottom_part])
print("Likelihood table for outlook: P(e | h) = P(Sunny | Yes):")
print("--------------------------------------------------------")
print(outlook_likelihood_df)
print()

# create and print the posterior probabilites for each class (i.e., yes and no)
posterior_probability = (outlook_likelihood_df.iloc[:-1, :] * outlook_likelihood_df.loc['Column Total']).div(outlook_likelihood_df['Row Total'], axis=0)
posterior_probability = posterior_probability.reindex(['Sunny', 'Overcast', 'Rainy'])
print("Posterior probability: P(h | e) = P(Yes | Sunny):")
print("-------------------------------------------------")
print(posterior_probability)
print()

# create label encoders
le_outlook = preprocessing.LabelEncoder()
le_play = preprocessing.LabelEncoder()

#Convert string labels into numbers
outlook_encoded=le_outlook.fit_transform(df['Outlook'])
play_encoded=le_play.fit_transform(df['Play Golf'])

# combine the data into single list of tuples
# most important for when multiple features are included
combined_features = list(zip(outlook_encoded))

# create a categorical classifier
# set alpha to 0 because we have already added laplace correction records
model = CategoricalNB(alpha=0)

# reshape data for training
outlook_encoded = outlook_encoded.reshape(-1,1)

# train the model using the training sets
model.fit(outlook_encoded, play_encoded)

print("Predictions based on weather outlook:")
print("-------------------------------------")
# for all categories of the feature variable, create predictions
for condition in le_outlook.classes_:
     print(f"Condition: {condition}")
     condition_transformed = le_outlook.transform([condition])
     yhat_prob = model.predict_proba([condition_transformed])
     prediction_statement = model.predict([condition_transformed])
     print(f"Prediction (i.e., will we play golf?): {le_play.inverse_transform(prediction_statement)}") 
     print(f"Probabilities: {yhat_prob}")
     print()



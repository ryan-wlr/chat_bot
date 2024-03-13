import pandas as pd
from transformers import pipeline
from copy import deepcopy

df = pd.read_csv('sentences.csv')

# Predict the price of the SalePrice variable
filepath = './llm-detect-ai-generated-text/train_essays.csv'
train_df = pd.read_csv(filepath, nrows=2)
print(train_df.head())

filepath2 = './llm-detect-ai-generated-text/test_essays.csv'
test_df = pd.read_csv(filepath2, nrows=2)
print(test_df.head())

# Import two columns from the training dataset
train_df_sub = deepcopy(train_df[['text']])

# Load classifier model
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

# Candidate labels
human_labels = ["human", "person"] 
ai_labels = ["AI", "computer", "bot", "machine"]

for index, row in train_df_sub.iterrows():
    sentence = row['text']
  
    # Get prediction
    prediction = classifier(sentence, candidate_labels=human_labels + ai_labels)
  
    # Print most likely label
    print(sentence)
    print("Predicted origin:", prediction['labels'][0])
    if prediction['labels'][0] == "human":
        dec = 0
    elif prediction['labels'][0] == "person":
        dec = .1
    elif prediction['labels'][0] == "AI":
        dec = 1
    elif prediction['labels'][0] == "computer":
        dec = .9
    elif prediction['labels'][0] == "bot":
        dec = .8
    elif prediction['labels'][0] == "machine":
        dec = .7
    print("decimal value: " + str(dec))
    train_df['generated']= dec

train_df[['id','generated']].to_csv('submission.csv', index=False)

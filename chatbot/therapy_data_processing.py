# scripts/preprocess.py

import pandas as pd
import re
import os

def clean_text(text):
    """
    Clean the input text by removing unwanted characters and normalizing whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_emotion_data():
    """
    Load and preprocess the emotion-emotion_69k.csv dataset.
    """
    # Define the path to the dataset
    data_path = os.path.join('data', 'emotion_emotion_69k.csv')

    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop NaN values
    df.dropna(inplace=True)

    # Check column names
    print(f"Columns in the dataset: {df.columns.tolist()}")

    # Assuming the dataset has columns 'content' and 'sentiment'
    text_column = 'content'
    emotion_column = 'sentiment'

    # Clean text data
    df['clean_text'] = df[text_column].apply(clean_text)

    # Encode emotions to numerical labels
    emotion_labels = df[emotion_column].unique()
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    df['label'] = df[emotion_column].map(emotion_to_id)

    print(f"Loaded and preprocessed {len(df)} samples from emotion_emotion_69k.csv")
    return df[['clean_text', 'label']], emotion_to_id

if __name__ == '__main__':
    # For testing purposes
    df, emotion_to_id = load_and_preprocess_emotion_data()
    print(df.head())

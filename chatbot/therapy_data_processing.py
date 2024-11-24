# scripts/preprocess.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Simplify text cleaning by converting to lowercase and stripping whitespace.
    """
    return str(text).lower().strip()

def load_and_preprocess_data():
    """
    Load and preprocess the emotion-emotion_69k.csv dataset.
    """
    # Define the path to the dataset
    data_path = os.path.join('dataset', 'emotion-emotion_69k.csv')

    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop NaN values
    df.dropna(subset=['text', 'emotion'], inplace=True)

    # Clean text data
    df['clean_text'] = df['text'].apply(clean_text)

    # Encode emotions to numerical labels
    emotion_labels = df['emotion'].unique()
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    df['label'] = df['emotion'].map(emotion_to_id)

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Save processed data
    train_df.to_csv(os.path.join('dataset', 'train.csv'), index=False)
    test_df.to_csv(os.path.join('dataset', 'test.csv'), index=False)

    # Save emotion labels mapping
    label_mapping = pd.DataFrame(list(emotion_to_id.items()), columns=['emotion', 'label'])
    label_mapping.to_csv(os.path.join('dataset', 'label_mapping.csv'), index=False)

    print("Data preprocessing completed and saved to the 'dataset/' directory.")

if __name__ == '__main__':
    load_and_preprocess_data()

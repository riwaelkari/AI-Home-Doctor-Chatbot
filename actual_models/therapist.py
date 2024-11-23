# scripts/train_emotion_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from ..chatbot.therapy_data_processing import load_and_preprocess_emotion_data
import joblib

def main():
    # Load and preprocess data
    emotion_df, emotion_to_id = load_and_preprocess_emotion_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        emotion_df['clean_text'], emotion_df['label'], test_size=0.2, random_state=42
    )

    # Create a pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', LogisticRegression(max_iter=2000)),
    ])

    # Train the model
    print("Training the emotion detection model...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotion_to_id.keys()))

    # Save the model
    model_path = os.path.join('models', 'emotion_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    # Save the emotion_to_id mapping
    mapping_path = os.path.join('models', 'emotion_to_id.joblib')
    joblib.dump(emotion_to_id, mapping_path)
    print(f"Emotion to ID mapping saved to {mapping_path}")

if __name__ == '__main__':
    main()

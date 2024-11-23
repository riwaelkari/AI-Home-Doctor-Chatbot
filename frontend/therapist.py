# scripts/chat_console.py

import joblib
import os
import re
import openai
from dotenv import load_dotenv
from chatbot.therapy_data_processing import clean_text

def load_models():
    """
    Load the emotion detection model and emotion-to-ID mapping.
    """
    model_path = os.path.join('..', 'models', 'emotion_model.joblib')
    mapping_path = os.path.join('..', 'models', 'emotion_to_id.joblib')

    # Load the emotion detection pipeline
    emotion_model = joblib.load(model_path)

    # Load the emotion to ID mapping
    emotion_to_id = joblib.load(mapping_path)
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    return emotion_model, id_to_emotion

def generate_response(user_input, emotion):
    """
    Generate a therapeutic response using OpenAI's GPT-3.5 Turbo model.
    """
    messages = [
        {"role": "system", "content": "You are a compassionate and skilled therapist dedicated to providing empathetic and supportive responses. Your goal is to help the user explore and understand their feelings in a safe and non-judgmental environment. If the user is unsure about their emotions, gently guide them with open-ended questions to help them articulate their feelings."},
        {"role": "user", "content": f"The user is feeling {emotion}. Provide a thoughtful, supportive, and empathetic response that encourages the user to share more about their emotions.\nUser: {user_input}\nTherapist:"}
    ]

    try:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None,
        )
        reply = response['choices'][0]['message']['content'].strip()
        return reply
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

def main():
    # Load environment variables
    load_dotenv()
    openai.api_key = os.getenv('SECRET_KEY')

    # Load models
    emotion_model, id_to_emotion = load_models()

    print("Therapist Bot is ready. Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Therapist: Remember, seeking professional help is always a good step. Take care!")
            break

        # Clean the input
        clean_input = clean_text(user_input)

        # Emotion Detection
        emotion_label = emotion_model.predict([clean_input])[0]
        emotion = id_to_emotion.get(emotion_label, 'neutral')

        # Generate Response
        response = generate_response(user_input, emotion)

        # Display the response
        print(f"Therapist: {response}\n")

if __name__ == '__main__':
    main()

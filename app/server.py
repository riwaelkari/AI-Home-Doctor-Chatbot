from flask import Flask, request, jsonify
import openai
import os
from train_models.neural_network import SymptomDiseaseModel
import logging
from flask_cors import CORS  # Add this for CORS handling
from data_processing import load_data, preprocess_data
import ast
from utils import encode_user_symptoms_fromgpt
import numpy as np
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)
# Set OpenAI API Key
openai.api_key = os.getenv('SECRET_TOKEN')  # Ensure the API key is set in your environment variables
y_train = training_data_cleaned['prognosis_encoded']
    # Initialize and train the model
model = SymptomDiseaseModel(y_train)
model.load_model('../models/saved_model.h5')
def gpt_response(message):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Change to your GPT model of choice
            messages=message,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return (response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in GPT response: {e}")
        return "I'm sorry, but I couldn't process your request at the moment."
def gpt_response_afterdisease(message):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Change to your GPT model of choice
            messages=message,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return (response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in GPT response: {e}")
        return "I'm sorry, but I couldn't process your request at the moment."
    
def decode_prediction(prediction, classes):
    predicted_index = np.argmax(prediction)
    predicted_disease = classes[predicted_index]
    return predicted_disease
def extract_list_from_string(input_str):
    """
    Extracts elements of the list whenever it detects '[' in the input string.
    If no list is detected, returns None.
    """
    start_index = input_str.find('[')
    end_index = input_str.find(']')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Extract the list part and split elements by commas
        list_str = input_str[start_index + 1:end_index]
        list_elements = [element.strip().strip("'").strip('"') for element in list_str.split(',')]
        return list_elements

    return None

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print(data['messages'])
        if not data or 'messages' not in data:
            logger.warning("No message provided in the request.")
            return jsonify({'error': 'No message provided.'}), 400
        user_message = data['messages']
        logger.info(f"Received message: {user_message}")

        # Initialize conversation history with the initial system prompt and user message
        messages = [
                {"role": "system", "content": "You are a helpful doctor chatbot. Your main role is extracting symptoms from the user and matching them with the list of symptoms i provide you, IF you get the symptoms return them as a list: ['symptom1','symptom2']. The list of symptoms is: " + ', '.join(all_symptoms)},
                {"role": "user", "content": user_message}
            ]

        # Get GPT response to extract symptoms
        gpt_reply = gpt_response(messages)
        logger.info("GPT response generated successfully.")
        user_symptoms = extract_list_from_string(gpt_reply)
        print(user_symptoms)

        if isinstance(user_symptoms, list):
            X_input = encode_user_symptoms_fromgpt(user_symptoms, all_symptoms)
            print(X_input)
            prediction = model.predict(X_input)
            predicted_disease = decode_prediction(prediction, classes)

            # Append the assistant's response to the conversation history
            messages.append({"role": "assistant", "content": gpt_reply})

            # Append the new system role instructing GPT to inform the user of the predicted disease
            messages.append({
                "role": "system",
                "content": f"You have diagnosed the patient with {predicted_disease}. Please inform the patient accordingly."
            })

            # Call GPT again to tell the user their predicted disease
            gpt_reply = gpt_response(messages)
        messages.append({"role": "user", "content":user_message})
        print(gpt_reply)
        return jsonify({
            'gpt_response': gpt_reply,
            'messages': messages
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
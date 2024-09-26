# server.py
from flask import Flask, request, jsonify
from data_processing import load_data, preprocess_data
from models.neural_network import SymptomDiseaseModel
import os
import openai
from utils import encode_user_symptoms, decode_prediction
import logging
from flask_cors import CORS  # Add this for CORS handling

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
openai.api_key = os.getenv('SECRET_TOKEN')  # Ensure the API key is set in your environment variables

try:
    # Load data and model once at startup
    symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
    training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)
    
    # Load the trained model
    y_train = training_data_cleaned['prognosis_encoded']
    model = SymptomDiseaseModel(y_train)
    model.load_model('models/saved_model.h5')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error during model loading: {e}")

def gpt_response(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a home doctor chatbot."},
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in GPT response: {e}")
        return "I'm sorry, but I couldn't process your request at the moment."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data or 'symptoms' not in data:
                logger.warning("No symptoms provided in the request.")
                return jsonify({'error': 'No symptoms provided.'}), 400
            
            user_symptoms = [sym.strip().lower() for sym in data['symptoms']]
            logger.info(f"Received symptoms: {user_symptoms}")
            
            # Encode symptoms
            X_input, unrecognized = encode_user_symptoms(user_symptoms, all_symptoms)
            
            if unrecognized:
                error_msg = f"Unrecognized symptoms: {', '.join(unrecognized)}"
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400

            # Make prediction using the model
            prediction = model.predict(X_input)
            predicted_disease = decode_prediction(prediction, classes)
            logger.info(f"Predicted disease: {predicted_disease}")
            
            # Get GPT response based on predicted disease
            gpt_message = f"I have predicted the disease: {predicted_disease}. Can you provide more details about it?"
            gpt_reply = gpt_response(gpt_message)
            logger.info("GPT response generated successfully.")
            
            return jsonify({
                'predicted_disease': predicted_disease,
                'gpt_response': gpt_reply
            }), 200
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

# app/server.py

from flask import Flask, render_template, request, jsonify
from models.neural_network import SymptomDiseaseModel
from data_processing import load_data, preprocess_data
import numpy as np
import os
from utils import encode_user_symptoms, decode_prediction,answer_question
import torch

app = Flask(__name__)

# Load the Label Encoder
symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)
# Load the trained model
y_train = training_data_cleaned['prognosis_encoded']
model = SymptomDiseaseModel(y_train)
model.load_model('models/saved_model.h5')

@app.route('/')
def index():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve symptoms from form
        user_input = request.form.get('symptoms')
        user_symptoms = [sym.strip().lower() for sym in user_input.split(',')]

        # Encode symptoms
        X_input, unrecognized = encode_user_symptoms(user_symptoms, all_symptoms)

        # Make prediction
        prediction = model.predict(X_input)
        predicted_disease = decode_prediction(prediction, classes)

        if unrecognized:
            warning = f"Warning: The following symptoms were not recognized: {', '.join(unrecognized)}. Please check the spelling or enter different symptoms."
        else:
            warning = None

        return render_template('result.html', disease=predicted_disease, warning=warning)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = answer_question(question)
            return render_template('ask_result.html', question=question, answer=answer)
    return render_template('ask.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided.'}), 400
        
        user_symptoms = [sym.strip().lower() for sym in data['symptoms']]
        
        # Encode symptoms
        X_input, unrecognized = encode_user_symptoms(user_symptoms, all_symptoms)
        
        # Make prediction
        prediction = model.predict(X_input)
        predicted_disease = decode_prediction(prediction, classes)
        
        response = {
            'predicted_disease': predicted_disease,
            'unrecognized_symptoms': unrecognized
        }
        
        return jsonify(response), 200

@app.route('/api/ask', methods=['POST'])
def api_ask():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided.'}), 400
        
        question = data['question']
        answer = answer_question(question)
        
        response = {
            'question': question,
            'answer': answer
        }
        
        return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
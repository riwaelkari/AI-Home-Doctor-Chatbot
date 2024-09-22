# app/server.py
from data_processing import load_data, preprocess_data
from flask import Flask, render_template, request
from models.neural_network import SymptomDiseaseModel
import pickle
import numpy as np
import os
app = Flask(__name__)

    # Load and preprocess data
symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
training_data_cleaned , testing_data_cleaned, classes, all_symptoms  = preprocess_data(symptom_df, testing_symptoms_df)
print(training_data_cleaned.shape)
print(testing_data_cleaned.shape)

X_train = training_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
y_train = training_data_cleaned['prognosis_encoded']
X_test = testing_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])
y_test = testing_data_cleaned['prognosis_encoded']

model = SymptomDiseaseModel(y_train)

model.train(X_train, y_train)
model.save_model()
# Load the Label Encoder

def encode_user_symptoms(user_symptoms, all_symptoms):
    """
    Converts user symptoms into a binary vector based on all possible symptoms.
    """
    input_vector = np.zeros(len(all_symptoms))
    symptom_to_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
    unrecognized = []

    for symptom in user_symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptom_to_index:
            index = symptom_to_index[symptom]
            input_vector[index] = 1
        else:
            unrecognized.append(symptom)

    if unrecognized:
        print(f"Warning: The following symptoms were not recognized: {', '.join(unrecognized)}")
        # Optionally, handle unrecognized symptoms more gracefully in the web app

    return input_vector.reshape(1, -1)

def decode_prediction(prediction, classes):
    """
    Converts the model's output into a disease name.
    """
    predicted_index = np.argmax(prediction)
    predicted_disease = classes[predicted_index]
    return predicted_disease

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
        X_input = encode_user_symptoms(user_symptoms, all_symptoms)

        # Make prediction
        prediction = model.predict(X_input)
        predicted_disease = decode_prediction(prediction, classes)

        return render_template('result.html', disease=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)

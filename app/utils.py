# app/utils.py

import numpy as np

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

    return input_vector.reshape(1, -1), unrecognized

def decode_prediction(prediction, classes):
    """
    Converts the model's output into a disease name.
    """
    predicted_index = np.argmax(prediction)
    predicted_disease = classes[predicted_index]
    return predicted_disease


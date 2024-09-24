# app/utils.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Initialize symptom encoding and decoding
def encode_user_symptoms(user_symptoms, all_symptoms):
    """
    Converts user symptoms into a binary vector based on all possible symptoms.
    Returns the encoded vector and a list of unrecognized symptoms.
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

    return input_vector.reshape(1, -1), unrecognized

def decode_prediction(prediction, classes):
    """
    Converts the model's output into a disease name.
    """
    predicted_index = np.argmax(prediction)
    predicted_disease = classes[predicted_index]
    return predicted_disease

# Initialize the Q&A pipeline using the fine-tuned Mixtral model
def initialize_qa_pipeline():
    model_path = "models/fine_tuned_mixtral"  # Path to the fine-tuned Mixtral model
    qa_pipeline = pipeline(
        "question-answering",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    return qa_pipeline

# Load the QA pipeline once
qa_pipeline = initialize_qa_pipeline()

def answer_question(question):
    """
    Uses the fine-tuned Mixtral Q&A pipeline to answer a medical question.
    """
    context = """
    Diabetes is a chronic health condition that affects how your body turns food into energy. 
    There are three main types of diabetes: type 1, type 2, and gestational diabetes. 
    Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, 
    presence of ketones in the urine, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections.
    """
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error in Q&A pipeline: {e}")
        return "I'm sorry, I couldn't understand your question. Please try again."

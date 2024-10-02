# app/utils.py
import torch
import numpy as np
import openai
import logging
logger = logging.getLogger(__name__)
import os
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

def encode_user_symptoms_fromgpt(user_symptoms, all_symptoms):
    """
    Encodes the user-extracted symptoms into a binary vector based on the list of all symptoms.
    
    :param user_symptoms: List of symptoms extracted from user input.
    :param all_symptoms: List of all possible symptoms.
    :return: Numpy array representing the encoded symptoms.
    """
    encoded = [1 if symptom.lower() in [s.lower() for s in user_symptoms] else 0 for symptom in all_symptoms]
    return np.array([encoded])

def decode_prediction(prediction, classes):
    """
    Converts the model's output into a disease name.
    """
    predicted_index = np.argmax(prediction)
    predicted_disease = classes[predicted_index]
    return predicted_disease


openai.api_key = os.getenv('SECRET_TOKEN')
def query_refiner(conversation, query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Updated to a valid model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user queries based on conversation context."},
            {"role": "user", "content": f"Given the following user message and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base, if the user input IS NOT a query RETURN NOTHING DO nnot generate a random question, use the keywords: describe, precautions, or severity in your formulated question.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

from langchain_openai import OpenAIEmbeddings

def find_match(input_text, index, faiss_store, top_k=2):
    """
    Finds the closest matches for the given input using the FAISS index.
    
    Args:
        input_text (str): The user input text to match.
        index (faiss.Index): The FAISS index used for similarity search.
        faiss_store (FAISS): The LangChain FAISS store containing the metadata.
        top_k (int): Number of top matches to retrieve.
    
    Returns:
        str: The combined metadata text from the top matches.
    """
    # Encode the input text to create an embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv('SECRET_TOKEN'))
    input_embedding = np.array([embeddings_model.embed_query(input_text)])  # Embed input

    # Search the FAISS index for the closest matches
    distances, indices = index.search(input_embedding, top_k)

    # Retrieve metadata from the FAISS store for the top matches
    matches = []
    for i in range(top_k):
        if indices[0][i] != -1:
            document = faiss_store.docstore.search(indices[0][i])
            if document and hasattr(document, 'page_content'):
                matches.append(document.page_content)

    # Combine the matches' content
    result = "\n".join(matches)
    return result

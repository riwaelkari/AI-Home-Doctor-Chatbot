# app/utils.py
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



def query_refiner(query, disease):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Ensure this is the desired model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that refines user queries based on the conversation context."
            },
            {
                "role": "user",
                "content": f"""
Given the following user message, formulate questions that would be most relevant to provide the user with an answer from a knowledge base. The refined question must:
User Query: 
{query}

Check user query for any of the three words [description/precautions/severity] and generate the question based on the following:

1. Use the format: "What is/are [description/precautions/severity] of {disease}?"
2. {disease} is the variable disease given to you.
3. If the user input is NOT a query, RETURN NOTHING. Do not generate a random question.

Ensure the keywords: "description", "precautions", or "severity" are used in your formulated question.


Refined Query:"""
            }
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

def query_refiner_severity(conversation, query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user queries based on the conversation context."},
            {"role": "user", "content": f"""
                Given the following user message and conversation log, formulate questions related to symptoms mentioned in the conversation. Each question must be in the format:

                "What is the severity of [symptom]?"

                Instructions:
                1. Identify all symptoms mentioned in the conversation log refering to the last query.
                2. Formulate one question for each symptom, asking about its severity.
                3. If there are no symptoms, RETURN NOTHING.

                CONVERSATION LOG: 
                {conversation}

                User Query: 
                {query}

                Refined Questions:"""}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Convert response to a list of questions
    questions = response.choices[0].message.content.splitlines()
    # Remove any empty strings in the list, if present
    questions = [q.strip() for q in questions if q.strip()]
    
    return questions




from langchain_openai import OpenAIEmbeddings

def find_match(input_text, embeddings_model, index, faiss_store, top_k=2):
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
    input_embedding = np.array([embeddings_model.embed_query(input_text)])  # Embed input
    print(input_embedding)
    # Search the FAISS index for the closest matches
    distances, indices = index.search(input_embedding, top_k)
    print(distances, indices)
    # Retrieve metadata from the FAISS store for the top matches
    matches = []
    for i in range(top_k):
        if indices[0][i] != -1:
            document = faiss_store.docstore.search(indices[0][i])
            if document and hasattr(document, 'page_content'):
                matches.append(document.page_content)
    print(matches)
    # Combine the matches' content
    result = "\n".join(matches)
    return result

def string_to_list(s):

    # Remove the square brackets if present
    s = s.strip('[]')

    # Split the string by commas
    items = s.split(',')

    # Strip whitespace and quotes from each item
    items = [item.strip(" '\"") for item in items]

    return items
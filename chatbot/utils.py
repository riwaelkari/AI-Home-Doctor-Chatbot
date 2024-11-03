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

# Set your OpenAI API key
openai.api_key = os.getenv('SECRET_TOKEN')

def query_refiner(query, disease):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Ensure this is the desired model
        messages=[
            {
                "role": "system",
                "content": f"""
You are a helpful assistant that refines user queries based on the conversation context.

**Instructions:**

- If the user's query **includes** any of the words **"description"**, **"precautions"**, or **"severity"**, generate a question in the format:

  **"What is/are [description/precautions/severity] of {disease}?"**

- Ensure that the keyword from the user's query (**"description"**, **"precautions"**, or **"severity"**) is used in your formulated question.

- **If the user's input does NOT include any of these keywords, respond with:**

  **"NO OUTPUT"**

- Do not generate any additional text or explanations.
"""
            },
            {
                "role": "user",
                "content": f"""
User Query:
{query}

Refined Query:"""
            }
        ],
        temperature=0,  # Set temperature to 0 for deterministic output
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content.strip()
    if output == "NO OUTPUT":
        return ""
    else:
        return output
import openai

def model_selector(query, list_of_models):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can change this to "gpt-4" if preferred
        messages=[
            {
                "role": "system",
                "content": f"""
You are an intelligent assistant designed to identify which model a user is referring to based on their query.

Instructions:

- Model 1: If the user's query mentions terms like "symptom disease model," "first model," or any other phrase that indicates Model 1, return `1`.

- Model 2: If the user's query mentions terms like "skin disease model," "image," "second model," or any other phrase that indicates Model 2, return `2`.

- Additional Guidelines:
  
  - Do not provide explanations or any additional text other than the specified outputs.
  
  - Ensure the response is either `1`, `2`, or `"NO OUTPUT"` based on the query.

**List of Models:**
{list_of_models}
"""
            },
            {
                "role": "user",
                "content": f"""
User Query:
{query}

Model Number:"""
            }
        ],
        temperature=0,  # Set to 0 for deterministic responses
        max_tokens=10,   # Sufficient for short outputs like "1", "2", or "NO OUTPUT"
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    output = response.choices[0].message.content.strip()
    
    if output == "NO OUTPUT":
        return ""
    elif output in {"1", "2"}:
        return int(output)
    else:
        # In case of unexpected output, you might want to handle it accordingly
        return ""

# Example Usage:
if __name__ == "__main__":
    list_of_models = """
1. Symptom Disease Model
2. Skin Disease Model
"""
    queries = [
        "Can you tell me more about the first model?",
        "I need information on the skin disease model.",
        "What is the weather today?"
    ]
    
    for q in queries:
        model_number = model_selector(q, list_of_models)
        print(f"Query: '{q}' => Model Number: {model_number}")

def query_refiner_severity(conversation, query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful assistant that refines user queries based on the conversation context.

**Instructions:**

- Identify all symptoms mentioned in the **most recent user message**.

- For each symptom, generate a question in the format:

  **"What is the severity of [symptom]?"**

- **If no symptoms are mentioned, respond with:**

  **"NO OUTPUT"**

- Do not generate any additional text or explanations.
"""
            },
            {
                "role": "user",
                "content": f"""
Conversation Log:
{conversation}

User Query:
{query}

Refined Questions:"""
            }
        ],
        temperature=0,  # Set temperature to 0 for deterministic output
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content.strip()
    if output == "NO OUTPUT":
        return []
    else:
        # Split the output into a list of questions
        questions = [line.strip() for line in output.split('\n') if line.strip()]
        return questions

def query_refiner_models(query, list_of_models):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Ensure this is the desired model
        messages=[
            {
                "role": "system",
                "content": f"""
You are a helpful assistant that refines user queries based on the conversation context.

**Instructions:**

- If the user's query is requesting **descriptions** of any model(s) from the provided list, generate questions in the format:

  **"What is the description of [model name]?"**

- Generate one question for each model the user is asking about.

- **If the user's input does NOT request descriptions, respond with:**

  **"NO OUTPUT"**

- Do not generate any additional text or explanations.

**List of Models:**
{list_of_models}
"""
            },
            {
                "role": "user",
                "content": f"""
User Query:
{query}

Refined Query:"""
            }
        ],
        temperature=0,  # Set temperature to 0 for deterministic output
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output =  response.choices[0].message.content.strip()
    if output == "NO OUTPUT":
        return ""
    else:
        return output

from langchain_community.embeddings import OpenAIEmbeddings

def find_match(input_text, embeddings_model, index, faiss_store, top_k=2):
    """
    Finds the closest matches for the given input using the FAISS index.
    
    Args:
        input_text (str): The user input text to match.
        embeddings_model: The embeddings model to use.
        index: The FAISS index used for similarity search.
        faiss_store: The LangChain FAISS store containing the metadata.
        top_k (int): Number of top matches to retrieve.
    
    Returns:
        str: The combined metadata text from the top matches.
    """
    # Encode the input text to create an embedding
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

def string_to_list(s):
    """
    Converts a string representation of a list into an actual list.

    Args:
        s (str): The string representation of the list.

    Returns:
        list: The converted list.
    """
    # Remove the square brackets if present
    s = s.strip('[]')

    # Split the string by commas
    items = s.split(',')

    # Strip whitespace and quotes from each item
    items = [item.strip(" '\"") for item in items]

    return items

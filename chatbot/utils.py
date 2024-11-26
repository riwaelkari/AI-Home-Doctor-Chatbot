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

Instructions:

- If the user's query includes any of the words "description", "precautions", or "severity", generate a question in the format:

  "What is/are [description/precautions/severity] of {disease}?"

- Ensure that the keyword from the user's query ("description", "precautions", or "severity") is used in your formulated question.

- If the user's input does NOT include any of these keywords, respond with:

  "NO OUTPUT"

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

def model_selector(conversation):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Change to "gpt-4" if preferred
        messages=[
            {
                "role": "system",
                "content": """
Based on the conversation, determine if the user is trying to choose on of the following models/doctors/secretary:
1. Model 1, Symptom disease doctor
2. Model2, Skin disease doctor
3. Model 3, Donna the secretary

Return the ONLY number of the model (1,2, or 3) as output if the user wants to choose, otherwise return NOTHING.
"""
            },
            {
                "role": "user",
                "content": f"""
Conversation:
{conversation}
"""
            }
        ],
        temperature=0,  # For deterministic responses
        max_tokens=10,   # Sufficient for "1", "2", "3", or "NOTHING"
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    output = response.choices[0].message.content.strip()
    
    if output == "NOTHING":
        return ""
    elif output in {"1", "2", "3"}:
        return int(output)
    else:
        # Handle unexpected output
        return ""


def query_refiner_severity(conversation, query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful assistant that refines user queries based on the conversation context.

Instructions:

- Identify all symptoms mentioned in the most recent user message.

- For each symptom, generate a question in the format:

  "What is the severity of [symptom]?"

- If no symptoms are mentioned, respond with:

  "NO OUTPUT"

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

Instructions:

- If the user's query is ONLY explicitely requesting descriptions of any model(s) from the provided list, generate questions in the format, otherwise generate NOTHING:

  "What is the description of [model name]?"

- Generate one question for each model the user is asking about.

NOTE:
- If the user's input does NOT request descriptions, respond with: NOTHING


List of Models:
{list_of_models}
"""
            },
            {
                "role": "user",
                "content": f"""
User Query:
{query}
"""
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
def guard_base(query):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this is the desired model
        messages=[
            {
                "role": "system", 
                "content": f""" 
            You are a helpful assistant responsible for determining if the user's query falls under allowed topics: normal conversation starters, or requests to delegate to doctors, or explanation or inqueries of what each doctor or secretary does who is donna, meaning, if the user talks about anything related to donna the secretary or the skin or symptom disease doctor, it asks normally but if it asks anything about something very far from its functionality, then it doesnt work. 
            if he asks you what you do also answer normally. 
              just if the topic is way off topic meaning then dont asnwer
             Normal conversation starters.
- Requests to delegate to doctors.
- Explanations or inquiries about what each doctor or secretary does or in other words what your resources do.
- Questions or discussions related to Donna, the secretary.
- Questions or discussions related to the skin or symptom disease doctor.
- Questions about what you (the assistant) do.
-If anything of the above have synonyms also answer normally.

In other words, if the user talks about anything related to Donna, the secretary, or the skin or symptom disease doctor, you should respond normally.

If the user asks what you do, you should also answer normally.

Instructions:

- If the query is allowed, respond with `'allowed'` only.
- If the query is not allowed, politely inform the user that you can only assist with medical-related inquiries, help delegate to available doctors, or explain what each one does.

            
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

    return output
def guard_base_symptom(query):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this is the desired model
        messages=[
            {
                "role": "system", 
                "content": f""" 

 You are a helpful assistant responsible for determining if the user's query falls under allowed topics:
   - Normal conversation starters.
   -Normal doctor patient interactions
   -Normal what the person is feeling in terms of wellness physical and anything that has symptoms 
- Medical questions related to symptoms or diseases, including:
  - Their descriptions
  - Precautions and prevention
  - Severity and progression
  - Causes and risk factors
  - Prognosis and outcomes
-
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

    return output
def guard_base_skin(query):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this is the desired model
        messages=[
            {
                "role": "system", 
                "content": f""" 
            You are a helpful assistant responsible for determining if the user's query falls under allowed topics:
   - Normal conversation starters.
   -Normal doctor patient interactions
   -Attach or picture  of skin  disease related inqueries
   -Normal what the person is feeling in terms of wellness physical and anything that has skin stuff 
- Medical questions related to skin diseases and infections, including:
  - A picture of it 
  - what it is
-
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

    return output
def guard_base_donna(query):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this is the desired model
        messages=[
            {
                "role": "system", 
                "content": f""" 
            You are a helpful assistant responsible for determining if the user's query falls under allowed topics.

Allowed Topics:
- Normal conversation starters.
- Requests to remind or schedule taking medications, can also mention the remind via email  or anything to do with timing reminders .
  - This includes reminding the user about specific medications, scheduling reminders, or answering general questions related to medications (e.g., dosage, timing).
  
Instructions:
- If the query is allowed, respond with `'allowed'` only.
- If the query is not allowed, politely inform the user that you can only assist with reminding or scheduling reminders to take medication.
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

    return output
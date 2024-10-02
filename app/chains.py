# chains.py
from data_processing import get_similar_docs
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils import encode_user_symptoms_fromgpt
from langchain_huggingface import HuggingFaceEmbeddings
from utils import query_refiner, find_match
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger(__name__)
class SymptomDiseaseChain:
    def __init__(self, all_symptoms, disease_model, classes, openai_api_key, faiss_store, faiss_index, embeddings_model):
        """
        Initializes the SymptomDiseaseChain with necessary components.

        Args:
            all_symptoms (list): List of all possible symptoms. 
            disease_model (object): Trained disease prediction model.
            classes (list): List of disease classes.
            openai_api_key (str): OpenAI API key.
        """
        self.all_symptoms = all_symptoms
        self.disease_model = disease_model
        self.classes = classes
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model="gpt-4o-mini"
        )
        self.prompt = self.get_symptom_extraction_prompt()
        self.response_prompt = self.get_response_generation_prompt()
        self.faiss_store = faiss_store  # Add FAISS store as an attribute
        self.faiss_index = faiss_index
        self.embeddings_model = embeddings_model
        self.get_info_prompt = self.return_info_prompt()
        

    def get_symptom_extraction_prompt(self):
        """
        Defines the prompt template for symptom extraction.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly medical assistant that wants to extract symptoms. Follow the steps below:

Possible symptoms (Confidential): {symptom_list}

User input: {user_input}

Conversation history: {conversation_history} (Do not explicitely mention what is said in history (such as User: ))

Steps:

1. First, check the **User input ONLY**:
   - If the user did not explicitly mention a symptom (such as asking for a description, precautions, severity, or general follow-up questions), **DO NOT PROCEED** to extract symptoms and ask the user if they are not feeling good to give you their symptoms.
   - If the user input is related to the existing diagnosis (e.g., asking about description, precautions, severity), respond accordingly, without adding any symptoms.


2. If the **User input** contains new symptoms:
   - Check the **conversation history** to see if the user previously mentioned symptoms. Only add these symptoms if they were previously mentioned and if the user is adding more symptoms.
   - Collect all symptoms, including both those in the current input and those in the conversation history, if applicable.

3. Return the ONLY the encoded symptoms (matching the possible symptoms list) as a comma-separated list ONLY if the user has provided new symptoms in their input and.
"""

        return PromptTemplate(
            input_variables=["user_input", "symptom_list","conversation_history"],
            template=template
        )

    def get_response_generation_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly and empathetic home doctor. Based on the conversation history, you should notify the user with the following disease {disease} and then tell him he could ask about the Description and precautions of the disease, and the severity of him symptoms.

Conversation history: {conversation_history}

User input: {user_input}

Do not give the user additional info about the disease.

Response:
"""
        return PromptTemplate(
            input_variables=["disease","conversation_history", "user_input"],
            template=template
        )
    def return_info_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly and empathetic home doctor. Based on the conversation history, you should give this info to the user asked for based on his symtoms and his disease (found in conversation history)

The info you should give: {info} if you think the info is not relevant to his question based on conversation history, tell him

Conversation history: {conversation_history}

User input: {user_input}

Only use the info you are given.

Response:
"""
        return PromptTemplate(
            input_variables=["info", "conversation_history", "user_input"],
            template=template
        )
    def extract_symptoms(self, user_input,conversation_history):
        """
        Extracts symptoms from the user input using the LLM.

        Args:
            user_input (str): The user's input message.

        Returns:
            list: A list of extracted symptoms.
        """#
        prompt_text = self.prompt.format(
            user_input=user_input,
            symptom_list=', '.join(self.all_symptoms),
            conversation_history = conversation_history
            )
        print(user_input)
        try:
            response = self.llm.invoke(prompt_text)
            logger.info(f"LLM Response for Symptom Extraction: {response}")
        except Exception as e:
            logger.error(f"Error while getting LLM response for symptom extraction: {e}")
            return []

        # Extract text from response
        if hasattr(response, 'content'):
            # For AIMessage objects
            text = response.content
        elif isinstance(response, dict):
            # For dictionary responses, if any
            text = response.get('content', '')
        elif isinstance(response, str):
            # For string responses
            text = response
        else:
            # Unexpected response type
            logger.error(f"Unexpected response type from LLM: {type(response)}")
            return []

        logger.info(f"Extracted Text: {text}")

        if text.strip().lower() == "no symptoms detected.":
            return []
        print(text)
        # Parse the response into a list of symptoms
        symptoms = [
        symptom.strip().lower()  # Normalize the symptoms

        for symptom in text.split(',')  # Split by comma to get individual symptoms
        ]
        print(symptoms)
        logger.info(f"Extracted Symptoms: {symptoms}")
        valid_symptoms = [symptom for symptom in symptoms if symptom in self.all_symptoms]

        return valid_symptoms

    def predict_disease(self, symptoms):
        """
        Predicts the disease based on the extracted symptoms.

        Args:
            symptoms (list): List of extracted symptoms.

        Returns:
            dict: A dictionary containing the predicted disease and extracted symptoms.
        """
        # Step 1: Encode Symptoms and Predict Disease
        X_input = encode_user_symptoms_fromgpt(symptoms, self.all_symptoms)
        prediction = self.disease_model.predict(X_input)
        predicted_disease = self.classes[np.argmax(prediction)]

        logger.info(f"Predicted Disease: {predicted_disease}")

        return {
            "predicted_disease": predicted_disease,
            "extracted_symptoms": symptoms
        }


    def generate_response(self, user_input, conv_history):
        """
        Generates a response based on user input by extracting symptoms and predicting disease.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.

        Returns:
            str: The chatbot's response.
            str or None: The predicted disease if available.
        """
        # Extract symptoms from user input
        symptoms = self.extract_symptoms(user_input,conv_history)
        refined_query = query_refiner(conv_history,user_input)
        print(refined_query)
        if symptoms:
            # Predict disease based on extracted symptoms
            prediction_result = self.predict_disease(symptoms)
            print("wrong loc")
            if "error" in prediction_result:
                response_message = prediction_result["error"]
                predicted_disease = None
            else:
                predicted_disease = prediction_result["predicted_disease"]
                print(type(conv_history))
                response_message = self.llm.invoke(self.response_prompt.format(
                    disease = predicted_disease,
                    conversation_history=conv_history,
                    user_input=user_input
                ))

                logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
        elif any(keyword in refined_query for keyword in ["description", "precautions", "severity"]):
            print("entered right location")
            similar_docs = find_match(refined_query, self.faiss_index, self.faiss_store)
            if similar_docs:
                info = similar_docs[0][0].page_content
                print(info)
            else:
                info = "No information available regarding your query."
            response_message = self.llm.invoke(self.get_info_prompt.format(
                    info = info,
                    conversation_history=conv_history,
                    user_input=user_input
            ))
            predicted_disease = None
        else:
            print("wrong loc 2")
            # No symptoms or asking about info is detected, generate a prompt to ask for symptoms
            response_message = self.llm.invoke(self.prompt.format(
                user_input=user_input,
                symptom_list= self.all_symptoms,
                conversation_history = conv_history
            )) 
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            predicted_disease = None
        return response_message.content, predicted_disease
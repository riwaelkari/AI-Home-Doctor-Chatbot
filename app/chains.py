# chains.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils import encode_user_symptoms_fromgpt
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger(__name__)

class SymptomDiseaseChain:
    def __init__(self, all_symptoms, disease_model, classes, openai_api_key):
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
            openai_api_key=openai_api_key
        )
        self.prompt = self.get_symptom_extraction_prompt()

    def get_symptom_extraction_prompt(self):
        """
        Defines the prompt template for symptom extraction.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a medical assistant. Extract the symptoms from the following user input based on the provided list of possible symptoms.

Possible symptoms: {symptom_list}

User input: {user_input}

Extracted symptoms (as a comma-separated list):
"""
        return PromptTemplate(
            input_variables=["user_input", "symptom_list"],
            template=template
        )

    def extract_symptoms(self, user_input):
        """
        Extracts symptoms from the user input using the LLM.

        Args:
            user_input (str): The user's input message.

        Returns:
            list: A list of extracted symptoms.
        """
        # Format the prompt with user input and symptom list
        prompt_text = self.prompt.format(
            user_input=user_input,
            symptom_list=', '.join(self.all_symptoms)
        )

        # Get the response from the LLM
        try:
            response = self.llm(prompt_text)
            logger.info(f"LLM Response: {response}")
        except Exception as e:
            logger.error(f"Error while getting LLM response: {e}")
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

        # Parse the response into a list of symptoms
        symptoms = [
            symptom.strip().lower()
            for symptom in text.split(',')
            if symptom.strip().lower() in [s.lower() for s in self.all_symptoms]
        ]

        logger.info(f"Extracted Symptoms: {symptoms}")

        return symptoms

    def predict_disease(self, user_input):
        """
        Predicts the disease based on the extracted symptoms.

        Args:
            user_input (str): The user's input message.

        Returns:
            dict: A dictionary containing the predicted disease and extracted symptoms.
        """
        # Step 1: Extract Symptoms
        symptoms = self.extract_symptoms(user_input)

        if not symptoms:
            return {
                "error": "No valid symptoms were extracted. Please try again with different symptoms."
            }

        # Step 2: Encode Symptoms and Predict Disease
        X_input = encode_user_symptoms_fromgpt(symptoms, self.all_symptoms)
        prediction = self.disease_model.predict(X_input)
        predicted_disease = self.classes[np.argmax(prediction)]

        logger.info(f"Predicted Disease: {predicted_disease}")

        return {
            "predicted_disease": predicted_disease,
            "extracted_symptoms": symptoms
        }

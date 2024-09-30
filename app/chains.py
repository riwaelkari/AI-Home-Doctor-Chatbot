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
        self.response_prompt = self.get_response_generation_prompt()

    def get_symptom_extraction_prompt(self):
        """
        Defines the prompt template for symptom extraction.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly medical assistant. Your goal is to help users by diagnosing potential diseases based on their symptoms so ask politely for their symptoms.

Possible symptoms: {symptom_list}

User input: {user_input}

Conversation history:
{conversation_history}
if the user mentioned symptoms in conversation history and is adding on them (the user said, I also have or etc), get the old symptoms and add them to the new symptoms to return the list
Extracted symptoms (as a comma-separated list).
"""
        return PromptTemplate(
            input_variables=["user_input", "symptom_list"],
            template=template
        )

    def get_response_generation_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly and empathetic home doctor. Based on the conversation history, you should notify the user with the following disease {disease}

Conversation history:
{conversation_history}

User input: {user_input}

Response:
"""
        return PromptTemplate(
            input_variables=["disease","conversation_history", "user_input"],
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

        # Parse the response into a list of symptoms
        symptoms = [
            symptom.strip().lower()
            for symptom in text.split(',')
            if symptom.strip().lower() in [s.lower() for s in self.all_symptoms]
        ]

        logger.info(f"Extracted Symptoms: {symptoms}")

        return symptoms

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

    def generate_response(self, user_input, conversation_history):
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
        symptoms = self.extract_symptoms(user_input)

        if symptoms:
            # Predict disease based on extracted symptoms
            prediction_result = self.predict_disease(symptoms)

            if "error" in prediction_result:
                response_message = prediction_result["error"]
                predicted_disease = None
            else:
                predicted_disease = prediction_result["predicted_disease"]
                extracted_symptoms = prediction_result["extracted_symptoms"]

                # Craft conversation history with diagnosis
                updated_history = f"{conversation_history}\nUser: {user_input}\nBot: You have been diagnosed with **{predicted_disease}** based on the symptoms: {', '.join(extracted_symptoms)}. If you have any other concerns or symptoms, feel free to share!"

                response_message = self.llm(self.response_prompt.format(
                    disease = predicted_disease,
                    conversation_history=updated_history,
                    user_input=user_input
                ))

                logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")

        else:
            # No symptoms detected, generate a prompt to ask for symptoms
            response_message = self.llm(self.prompt.format(
                conversation_history=conversation_history,
                user_input=user_input,
                symptom_list= self.all_symptoms
            ))
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            predicted_disease = None
        print(response_message)
        return response_message.content, predicted_disease
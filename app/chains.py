# chains.py
from data_processing import get_similar_docs
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils import encode_user_symptoms_fromgpt
from langchain_huggingface import HuggingFaceEmbeddings
from utils import query_refiner, predict_disease, extract_symptoms
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger(__name__)
class SymptomDiseaseChain:
    def __init__(self, all_symptoms, disease_model, classes, openai_api_key):#, faiss_store):
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
       #self.faiss_store = faiss_store  # Add FAISS store as an attribute
        #self.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #self.get_info_prompt = self.return_info_prompt()

    def get_symptom_extraction_prompt(self):
        """
        Defines the prompt template for symptom extraction.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly medical assistant. When the user first talksa to you ask the user if they are feeling unwell to give you their symptoms. 

Possible symptoms (Confidential): {symptom_list}

User input: {user_input}

Conversation history: {conversation_history}

Return Extracted symptoms and encode them to match the list and return them (as a comma-separated list).
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
You are a friendly and empathetic home doctor. Based on the conversation history, you should give this info to the user asked for based on his symtoms and his disease {predicted_disease}

The info you should give: {info} if you think the info is not relevant to his question based on conversation history, tell him

Conversation history: {conversation_history}

User input: {user_input}

Response:
"""
        return PromptTemplate(
            input_variables=["disease", "info", "conversation_history", "user_input"],
            template=template
        )
    

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
        symptoms = extract_symptoms(user_input,conv_history)
        """
        refined_query = query_refiner(conv_history,user_input)
        if any(keyword in refined_query for keyword in ["description", "precautions", "severity"]):
            similar_docs = get_similar_docs(refined_query, self.embeddings_model, self.faiss_store.index, split_documents=self.faiss_store.documents)
            if similar_docs:
                info = similar_docs[0][0].page_content
            else:
                info = "No information available regarding your query."
            response_message = self.llm(self.prompt.format(
                    info = info,
                    disease = predicted_disease,
                    conversation_history=conv_history,
                    user_input=user_input
            )) 
            """
        if symptoms:
            # Predict disease based on extracted symptoms
            prediction_result = predict_disease(symptoms)

            if "error" in prediction_result:
                response_message = prediction_result["error"]
                predicted_disease = None
            else:
                predicted_disease = prediction_result["predicted_disease"]
                print(type(conv_history))
                response_message = self.llm(self.response_prompt.format(
                    disease = predicted_disease,
                    conversation_history=conv_history,
                    user_input=user_input
                ))

                logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
        else:
            # No symptoms detected, generate a prompt to ask for symptoms
            response_message = self.llm(self.prompt.format(
                user_input=user_input,
                symptom_list= self.all_symptoms,
                conversation_history = conv_history
            )) 
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            predicted_disease = None
        return response_message.content, predicted_disease
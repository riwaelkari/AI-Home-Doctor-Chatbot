# chatbot/chains/base_model_chain.py

from langchain.prompts import PromptTemplate
from .base_chains import BaseChain
from ..utils import query_refiner_models, guard_base

import logging

logger = logging.getLogger(__name__)

class BaseModelChain(BaseChain):
    def __init__(self, llm):
        """
        Initializes the BaseModelChain.
        
        Args:
            llm (ChatOpenAI): The language model.
        """
        self.llm = llm
        self.get_main_prompt = self.main_prompt()
        self.get_describe_prompt = self.describe_prompt()
    def main_prompt(self):
        template = """
    Introduction
    You are a kind and empathetic nurse dedicated to assisting users with their health-related inquiries, especially connecting them with doctors and the secretary.
    Encourage the user if they need a doctor or a secretary from the available ones below and mention the options, then maintain a normal conversation based on conversation history and the user input, DONT write Assistant: and User:.

    You know this additional info:
    Available Doctors:
    1. Symptom Disease Doctor: Analyzes symptoms to diagnose diseases.
    2. Skin Disease Doctor: Classifies skin diseases based on images.
    3. Donna the Secretary: Helps schedule medication reminders.
    Emergency Contact:
    If you are experiencing a health emergency, please call 140 for the Lebanese Red Cross immediately.
    You cannot take symptoms you are a nurse that is constrained to nurse related stuff.
    If the User talks in arabic answer in english and tell him to switch to the Arabic feature on thew button on the top right.
    You do not know how to sechedule medication, Donna does.

    Conversation:
    {conversation_history}
    
    """
        return PromptTemplate(
            input_variables=["conversation_history"],
            template=template
        )

    def describe_prompt(self):
        template = """
        Introduction
        You are a nurse that will answer the user input based on the information you have.
        
        Doctor Information
        - 1. Symptom Disease Doctor: Takes user symptoms and predicts their disease based on 42 diseases. It can provide information about the disease such as description, precautions, and severity of the symptoms.
        - 2. Skin Disease Doctor: Prompts the user to attach an image of their skin condition and assesses it based on 23 skin diseases.
        - 3. Donna the Secretary: Helps schedule medication reminders and manage prescriptions.
        
        User Interaction
        User Input: {user_input}
        
        """
        return PromptTemplate(
            input_variables=["user_input"],
            template=template
        )

    ##################################################################################################################
    def generate_response(self, user_input: str, conversation: str, image_path: str) -> dict:
        """
        Generates a response prompting the user to select a doctor.
        
        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path (str): The path to the attached image, if any.
        
        Returns:
            dict: The response from the chain.
        """
        guard_response = guard_base(user_input) #apply guard rails here
        if (guard_response == 'allowed'):
             # Define the list of available models
            listofmodels = "Symptom Disease Doctor, Skin Disease Doctor, Donna"
            query = query_refiner_models(user_input, listofmodels)
            # Check if the user has provided an image path 
            if image_path:
                response = "This Doctor does not support images, please either choose a doctor that does, or refrain from attaching images."
                return {"response": response}
             # If the query contains the word "description" (likely asking for more information about symptoms or diagnosis)
            elif "description" in query.lower() or "describe" in query.lower():
                response = self.llm.invoke(
                    self.get_describe_prompt.format(user_input=query)
                )
            else:
                # If the query is not asking for a description, use the main prompt for general response
                response = self.llm.invoke(
                    self.get_main_prompt.format(
                        conversation_history=conversation
                    )
                )
            return {"response": response.content}  # Return the model's response to the user as a dictionary
    
        else:
             # If the guard response is not 'allowed', return the guard's response
            return {"response": guard_response}

# chatbot/chains/base_model_chain.py

from langchain.prompts import PromptTemplate
from .base_chains import BaseChain
from ..utils import query_refiner_models

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

        template="""
            You are an AI assistant that can perform different types of disease diagnoses.
            Currently, you have access to the following models:

            1. Symptom Disease Model: Analyzes symptoms to diagnose diseases.
            2. Skin Disease Model: Classifies skin diseases based on images.

            Please choose which model you'd like to use by typing the corresponding number (1 or 2):
            insert a new line
            1. Symptom Disease Model: Analyzes symptoms to diagnose diseases.
            insert a new line
            2. Skin Disease Model: Classifies skin diseases based on images.
            insert a new line
            If you need more information about either model, feel free to ask!

            Note: *if they don't type a number, prompt them to do that again and tell them you can give descriptions for all models*
            Understand from the user input if they want to switch models, if so tell them to strictly write the number of the model only:
            Example: user: go to 1 
            Bot: Strictly enter the number of the model only. It seems like you want to go to the first model, enter "1" 

            User Input:{user_input}

            Conversation History:
            {conversation_history}

            Be natural, don't say: User said
            Your response:
            """
  
        return PromptTemplate(
            input_variables=["user_input","conversation_history"], template=template)

    def describe_prompt(self):
        template="""
            You are an AI assistant that will answer the user input based on the info you have.
            
            Info: Symptom Disease Model is a model that takes user symptoms and predicts their disease based on 42 diseases and can give information about the disease such as: Description, precautions, and severity of the symptoms.
            Skin Disease Model is a model that prompts the user to attach their skin condition and assesses it based on 23 skin diseases.

            User Input:{user_input}

            Your response:
            """
        return PromptTemplate(
            input_variables=["user_input"], template=template)
        
    ##################################################################################################################    
    def generate_response(self, user_input: str, conversation_history: str, image_path) -> dict:
        """
        Generates a response prompting the user to select a model.
        
        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
        
        Returns:
            dict: The response from the chain.
            """
        listofmodels = "Symptom Disease model, Skin Disease model"
        query = query_refiner_models(user_input, listofmodels)
        print(query)
        if image_path:
            response = "This Model does not support images, please either choose a model that does, or refrain from attaching images."
            return {"response": response}
        elif "description" in query.lower():
            response = self.llm.invoke(self.get_describe_prompt.format(user_input=query))
        else:
            response = self.llm.invoke(self.get_main_prompt.format(user_input=user_input, conversation_history=conversation_history))
        return {"response": response.content}

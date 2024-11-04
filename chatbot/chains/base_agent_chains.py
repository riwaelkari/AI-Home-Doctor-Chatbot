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
        template = """
        ### Introduction
        You are an empathetic and kind nurse that will guide the user to different doctor models. YOU ARE NOT A doctor. YOU CANNOT diagnose anything youself, you can only guide the user to choose a model.

        ### Instructions
        You are an empathetic and kind nurse that will guide the user to different doctor models. YOU ARE NOT A doctor. YOU CANNOT diagnose anything youself, you can only guide the user to choose a model.

        -Greet the user warmly as a nurse and tell them to choose a number (1 or 2) corresponding to the Doctors below
        
        1. Symptom Disease Model: Analyzes symptoms to diagnose diseases.
        2. Skin Disease Model: Classifies skin diseases based on images.
        
        -You have access to the description of each model so you can ask them if they want descriptions.
        
        -maintain a natural conversation flow AT ALL COSTS.

        **Note:** *If the user doesn't type a number, prompt them to do that again and inform them that you can provide descriptions for all models.*
        Understand from the user input if they want to switch models. If so, instruct them to strictly write the number of the model only.
        
        **Example:**
        - **User:** 
        
        - **Bot:** Strictly enter the number of the model only. It seems like you want to go to the first model. Please enter "1".
        
        ### User Interaction
        **User Input:** {user_input}
        
        **Conversation History:**
        {conversation_history}
        
        Be natural; don't say: "User said"
        
        **Your response:**
        """

        return PromptTemplate(
            input_variables=["user_input", "conversation_history"],
            template=template
        )

    def describe_prompt(self):
        template = """
        ### Introduction
        You are a nurse that will answer the user input based on the information you have.
        
        ### Model Information
        - **Symptom Disease Model:** Takes user symptoms and predicts their disease based on 42 diseases. It can provide information about the disease such as description, precautions, and severity of the symptoms.
        - **Skin Disease Model:** Prompts the user to attach an image of their skin condition and assesses it based on 23 skin diseases.
        
        ### User Interaction
        **User Input:** {user_input}
        
        **Your response:**
        """
        return PromptTemplate(
            input_variables=["user_input"],
            template=template
        )

    ##################################################################################################################
    def generate_response(self, user_input: str, conversation_history: str, image_path: str) -> dict:
        """
        Generates a response prompting the user to select a model.
        
        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path (str): The path to the attached image, if any.
        
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
            response = self.llm.invoke(
                self.get_describe_prompt.format(user_input=query)
            )
        else:
            response = self.llm.invoke(
                self.get_main_prompt.format(
                    user_input=user_input,
                    conversation_history=conversation_history
                )
            )
        return {"response": response.content}

# agent.py
#working agent
import os
from langchain_openai import ChatOpenAI
from PIL import Image
import io
import logging
from .chains.base_chains import BaseChain
from .utils import model_selector

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.chains = {}
        self.default_chain = None
        self.awaiting_input = None
        self.current_chain = None
        self.current_chain_name = 'Nurse'  # Initialize bot name
        openai_api_key = os.getenv('SECRET_TOKEN')
        # Initialize the LLM for translation
        self.translation_llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",  # Replace with your desired model
        openai_api_key=openai_api_key
    )



    def translate_text(self, text: str, target_language: str) -> str:
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        translation = self.translation_llm(prompt)
        return translation.content.strip()


    def register_chain(self, name: str, chain: BaseChain):
        """
        Registers a new chain with the agent.

        Args:
            name (str): The name identifier for the chain.
            chain (BaseChain): An instance of a chain inheriting from BaseChain.
        """
        if not isinstance(chain, BaseChain):
            raise ValueError(f"The chain must inherit from BaseChain. Provided chain '{name}' does not.")
        self.chains[name] = chain
        logger.info(f"Registered chain: {name}")

    def set_default_chain(self, chain: BaseChain):
        """
        Sets the default chain to use if no specific chain is matched.

        Args:
            chain (BaseChain): An instance of a chain inheriting from BaseChain.
        """
        if not isinstance(chain, BaseChain):
            raise ValueError("The default chain must inherit from BaseChain.")
        self.default_chain = chain
        logger.info("Default chain set.")

    def determine_chain(self, user_input: str) -> BaseChain:
        """
        Determines which chain to use based on user input.

        Args:
            user_input (str): The user's input message.

        Returns:
            BaseChain: The selected chain based on the input.
        """
        determine = model_selector(user_input)
        # Check if the user is selecting a chain
        if determine == 1:
            self.current_chain = self.chains.get('symptom_disease', self.default_chain)
            self.current_chain_name = 'Symptom Disease Doctor'  # Update bot name
            logger.info("Switched to chain: symptom_disease")
            user_input = "Hi"
            return self.current_chain
        elif determine == 2:
            self.current_chain = self.chains.get('skin_disease', self.default_chain)
            self.current_chain_name = 'Skin Disease Doctor'  # Update bot name
            logger.info("Switched to chain: skin_disease")
            user_input = "Hi"
            return self.current_chain
        elif determine == 3:
            self.current_chain = self.chains.get('donna', self.default_chain)
            self.current_chain_name = 'Donna'  # Update bot name
            logger.info("Switched to chain: donna")
            user_input = "Hi"
            return self.current_chain
        elif user_input.lower() == "reset":
            self.current_chain = None
            self.current_chain_name = 'Nurse'  # Reset bot name
            logger.info("Reset to default chain")
            return self.default_chain

        # Use the current chain if one is set
        if self.current_chain:
            return self.current_chain

        # Default to the base chain
        self.current_chain_name = 'Nurse'  # Ensure default name
        return self.default_chain


    def handle_request(self, user_input: str, conversation_history: str, image_path, language='En') -> dict:
        """
        Handles the user request by delegating to the appropriate chain.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path: Path to the uploaded image, if any.
            language (str): The language selected by the user ('En' or 'Ar').

        Returns:
            dict: The response from the selected chain.
        """
        logger.info(f"user input:" + user_input)
        chain = self.determine_chain(user_input)
        if not chain:
            logger.error("No chain available to handle the request.")
            return {
                'response': "I'm sorry, I didn't understand that. Could you please rephrase?",
                'bot_name': self.current_chain_name,
                'bot_icon': 'images/nurse_icon.png'  # Default icon
            }

        logger.info(f"Delegating to chain: {chain.__class__.__name__}")

        if user_input.lower() == 'reset':
            user_input = 'Hi'

        # Generate the response using the chain
        response = chain.generate_response(user_input, conversation_history, image_path)
        response['bot_name'] = self.current_chain_name

        # If language is Arabic, translate the response back to Arabic
        if language == 'Ar':
            response_in_arabic = self.translate_text(response['response'], 'Arabic')
            print(f"Translated response to Arabic: {response_in_arabic}")
            response['response'] = response_in_arabic

        # Include bot icon based on current chain
        if self.current_chain_name == 'Symptom Disease Doctor':
            response['bot_icon'] = 'images/symptom_disease_icon.png'
        elif self.current_chain_name == 'Skin Disease Doctor':
            response['bot_icon'] = 'images/skin_disease_icon.png'
        elif self.current_chain_name == 'Donna':
            response['bot_icon'] = 'images/donna_icon.png'
        else:
            response['bot_icon'] = 'images/nurse_icon.png'
        print(response['bot_icon'])
        return response
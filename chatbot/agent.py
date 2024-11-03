# chatbot/agent.py
from PIL import Image
import io
import logging
from .chains.base_chains import BaseChain

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.chains = {}
        self.default_chain = None
        self.awaiting_input = None
        self.current_chain = None  # Add this line

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
        # Check if the user is selecting a chain
        if user_input == "1":
            self.current_chain = self.chains.get('symptom_disease', self.default_chain)
            logger.info("Switched to chain: symptom_disease")
            return self.current_chain
        elif user_input == "2":
            self.current_chain = self.chains.get('skin_disease', self.default_chain)
            logger.info("Switched to chain: skin_disease")
            return self.current_chain
        elif user_input.lower() == "reset":
            self.current_chain = None
            logger.info("Reset to default chain")
            return self.default_chain

        # Use the current chain if one is set
        if self.current_chain:
            return self.current_chain

      
        # Default to the base chain
        return self.default_chain

    def handle_request(self, user_input: str, conversation_history: str, image_path) -> dict:
        """
        Handles the user request by delegating to the appropriate chain.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.

        Returns:
            dict: The response from the selected chain.
        """
        chain = self.determine_chain(user_input)
        if not chain:
            logger.error("No chain available to handle the request.")
            return {'response': "I'm sorry, I didn't understand that. Could you please rephrase?"}
        #print("helooooo")

        logger.info(f"Delegating to chain: {chain.__class__.__name__}")
        
        try:
            response = chain.generate_response(user_input, conversation_history,image_path)
            print(response)
            return response
        except Exception as e:
            logger.error(f"Error in chain '{chain.__class__.__name__}': {e}", exc_info=True)
            return {'response': "An error occurred while processing your request. Please try again later."}
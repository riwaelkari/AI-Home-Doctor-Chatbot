# chatbot/agent.py

import logging
from .chains.base_chains import BaseChain

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.chains = {}
        self.default_chain = None

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
        # Simple keyword-based routing; can be enhanced with NLP techniques
        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in ['symptom', 'disease', 'diagnose']):
            return self.chains.get('symptom_disease', self.default_chain)
        elif any(keyword in user_input_lower for keyword in ['info', 'information']):
            return self.chains.get('information', self.default_chain)
        # Add more conditions for other chains as needed

        # If no specific chain is matched, return the default chain
        return self.default_chain

    def handle_request(self, user_input: str, conversation_history: str) -> dict:
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

        logger.info(f"Delegating to chain: {chain.__class__.__name__}")
        try:
            response = chain.generate_response(user_input, conversation_history)
            return response
        except Exception as e:
            logger.error(f"Error in chain '{chain.__class__.__name__}': {e}", exc_info=True)
            return {'response': "An error occurred while processing your request. Please try again later."}
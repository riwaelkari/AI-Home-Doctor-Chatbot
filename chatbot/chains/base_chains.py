# chatbot/chains/base_chains.py

from abc import ABC, abstractmethod

class BaseChain(ABC):
    """
    Abstract base class for creating different types of response generation chains in a chatbot.

    This class defines a common interface for all response generation chains that process user input
    and conversation history to produce a response. It ensures that all subclasses implement the `generate_response` method,
    which is responsible for generating a response based on user input, conversation history, and any additional data (e.g., images).

    Methods:
        generate_response(user_input: str, conversation_history: str, image: Optional[any]) -> dict:
            Abstract method to be implemented by subclasses for generating responses.
    """
    @abstractmethod
    def generate_response(self, user_input: str, conversation_history: str,image) -> dict:
        pass
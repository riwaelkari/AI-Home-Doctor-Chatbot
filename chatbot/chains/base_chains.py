from abc import ABC, abstractmethod

class BaseChain(ABC):
    @abstractmethod
    def generate_response(self, user_input: str, conversation_history: str) -> dict:
        """
        Generate a response based on user input and conversation history.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.

        Returns:
            dict: A dictionary containing at least a 'response' key and optionally additional data.
                  Example:
                  {
                      'response': "Your response message here.",
                      'predicted_disease': "Disease Name"  # Optional
                  }
        """
        pass
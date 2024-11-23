# chatbot/chains/base_chains.py

from abc import ABC, abstractmethod

class BaseChain(ABC):
    @abstractmethod
    def generate_response(self, user_input: str, conversation_history: str,image) -> dict:
        pass
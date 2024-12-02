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
    """
    A class that manages conversation chains, interacts with OpenAI's GPT model for text translation, 
    and provides functionality for chatbot-based tasks.

    This class is designed to handle a variety of conversational states and processes, including:
    - Storing and managing conversation chains.
    - Handling user input and switching between conversation chains.
    - Using OpenAI's GPT-3.5-turbo model to translate text between languages.
    
    Attributes:
        chains (dict): A dictionary that stores different conversation chains or interaction contexts.
        default_chain (Optional[BaseChain]): The default conversation chain to be used when no other chain is active.
        awaiting_input (Optional[str]): A placeholder that tracks if the bot is awaiting user input.
        current_chain (Optional[BaseChain]): The current active conversation chain.
        current_chain_name (str): The name of the current conversation chain (default is 'Nurse').
        translation_llm (ChatOpenAI): The language model used to translate text between languages.
    
    Methods:
        __init__: Initializes the class with default values for conversation management and sets up
                  the OpenAI language model for text translation.
    """
    def __init__(self):
        """
        Initializes the class with default values for conversation management and sets up
        the OpenAI language model (GPT-3.5-turbo) for text translation.
        
        This constructor sets up necessary variables for tracking conversation chains,
        and initializes a translation model using OpenAI's GPT-3.5-turbo with an API key fetched
        from the environment variable 'SECRET_TOKEN'.
        
        Attributes initialized:
            chains: An empty dictionary to store conversation chains.
            default_chain: Set to None, as no default chain is specified initially.
            awaiting_input: Set to None, used to check if the bot is waiting for user input.
            current_chain: Set to None, used to track the current active conversation chain.
            current_chain_name: Set to 'Nurse' by default, representing the current bot's name.
            translation_llm: Initialized with ChatOpenAI for translation purposes, using GPT-3.5-turbo.
        
        Raises:
            KeyError: If the 'SECRET_TOKEN' environment variable is not found.
        """
        self.chains = {}   # Dictionary to store different chains of conversation or interactions.
        self.default_chain = None  # Placeholder for the default conversation chain.
        self.awaiting_input = None  # Placeholder for tracking whether the system is awaiting user input.
        self.current_chain = None # Placeholder for the current active chain of conversation.
        self.current_chain_name = 'Nurse'  # Initialize bot name to 'Nurse' by default.
        openai_api_key = os.getenv('SECRET_TOKEN') # Retrieves the OpenAI API key from the environment variables.
        # Initialize the LLM for translation 
        self.translation_llm = ChatOpenAI(
        temperature=0.7,  # The temperature parameter controls the randomness of the model's output.
        model_name="gpt-3.5-turbo",  # Specifies the model used for translation (GPT-3.5 turbo)
        openai_api_key=openai_api_key # Provides the API key for OpenAI's service.
    )

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translates the provided text into the specified target language using OpenAI's GPT-3.5-turbo model.
        
        This method constructs a prompt instructing the model to translate the given text to the target language
        and returns the translated content. The model is used with a temperature setting of 0.7 to control
        the randomness of the output.
        
        Args:
            text (str): The input text that needs to be translated.
            target_language (str): The language to which the text should be translated.
        
        Returns:
            str: The translated version of the input text.
        """
         # Constructs the prompt to instruct the translation model.
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
         # Uses the LLM to generate the translation.
        translation = self.translation_llm(prompt)
        # Returns the translated text, stripping any leading/trailing whitespace.
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
    def set_nurse_chain(self, chain: BaseChain):
        """
        Sets the default chain to use if no specific chain is matched.

        Args:
            chain (BaseChain): An instance of a chain inheriting from BaseChain.
        """
        if not isinstance(chain, BaseChain):
            raise ValueError("The default chain must inherit from BaseChain.")
        self.current_chain = chain
        self.current_chain_name = 'Nurse'  # Initialize bot name
        logger.info("Current chain set.")    

    def determine_chain(self, user_input: str,conversation: str) -> BaseChain:
        """
        Determines which chain to use based on user input.

        Args:
            user_input (str): The user's input message.

        Returns:
            BaseChain: The selected chain based on the input.
        """
        determine = model_selector(conversation)
        print(determine)
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
       # elif user_input.lower() == "reset":
        #    self.current_chain = None
         #   self.current_chain_name = 'Nurse'  # Reset bot name
        #    logger.info("Reset to default chain")
        #    return self.default_chain

        # Use the current chain if one is set
        if self.current_chain:
            return self.current_chain

        # Default to the base chain
        self.current_chain_name = 'Nurse'  # Ensure default name
        return self.default_chain


    def handle_request(self, user_input: str, conversation_history: str, image_path, language='En', reset=False) -> dict:
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
        logger.info(f"Conv History:  {conversation_history}")
        logger.info(f"user input: {user_input}")
        conversation_history = conversation_history +"\n"+ "Patient: " + user_input
        chain = self.determine_chain(user_input,conversation_history)
        if not chain:
            logger.error("No chain available to handle the request.")
            return {
                'response': "I'm sorry, I didn't understand that. Could you please rephrase?",
                'bot_name': self.current_chain_name,
                'bot_icon': 'images/nurse_icon.png'  # Default icon
            }

        logger.info(f"Delegating to chain: {chain.__class__.__name__}")

      #  if user_input.lower() == 'reset':
        #    user_input = 'Hi'

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
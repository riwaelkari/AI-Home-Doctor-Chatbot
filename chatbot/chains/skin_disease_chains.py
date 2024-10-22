# chatbot/chains/skin_disease_chains.py

from .base_chains import BaseChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from PIL import Image
import io
from actual_models.skin_disease_model import process_image
logger = logging.getLogger(__name__)

class SkinDiseaseChain(BaseChain):
    def __init__(self, disease_model, llm):
        """
        Initializes the SkinDiseaseChain.

        Args:
            disease_model (SkinDiseaseModel): The skin disease classification model.
            llm (ChatOpenAI): The language model.
        """
        self.disease_model = disease_model
        self.llm = llm
        self.get_main_prompt = self.main_prompt()
        self.get_disease = self.return_disease()
        self.awaiting_image = False  # Flag to track if the chain is waiting for an image

    def main_prompt(self):
        """
        Creates the LLMChain with a specific prompt template.

        Returns:
            LLMChain: The initialized chain.
        """
        template =  """
You are a friendly medical assistant at home specialized in diagnosing skin diseases based on images. Interact with the user {user_input} and mainly Ask him to give you a picture of his skin issue by asking him to attach the picture using the attach button on the left side of the text box.

Conversation history: {conversation_history}

Answer based on Conversation history also.

Be natural, dont say: User said
"""
        
        return PromptTemplate(
            input_variables=["user_input","conversation_history"],template= template)
    
    def return_disease(self):
        """
        Creates a prompt template for providing information based on the predicted disease.

        Returns:
            LLMChain: The initialized chain with the disease-specific prompt.
        """
        template = """
You are a friendly home doctor. Inform the User with his disease results without mentioning numbers: {disease}
*Conversation History:* 
{conversation_history}

*User Input:* 
{user_input}


Be natural, dont say: User said

"""
        return PromptTemplate(
            input_variables=["user_input","conversation_history"],template= template)

    
 ##############################################################################################################

    def extract_disease_from_image(self, image_bytes, device='cpu'):
        """
    Converts image bytes to RGB and predicts the disease using the disease model.

    Args:
        image_bytes (bytes): The image data in bytes.
        device (str, optional): Device to perform computation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        str or None: The predicted disease if successful, else None.
       """
    
    # Step 1: Convert bytes to PIL Image and ensure it's in RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info("Image successfully converted to RGB.")

        # Step 2: Predict the disease using the classifier's `predict` method
        # Assuming `predict` handles preprocessing internally
        predicted_disease = self.disease_model.predict(img, device=device)
        
        return predicted_disease

##########################################################################################################################


    def generate_response(self, user_input: str, conversation_history: str, image_bytes=None, device='cpu') -> dict:
        """
        Generates a response based on user input, conversation history, and an optional image.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_bytes (bytes, optional): An image related to skin conditions, if provided.
            device (str, optional): Device to perform computation ('cpu' or 'cuda').

        Returns:
            dict: A dictionary containing the chatbot's response and any additional data.
        """
        predicted_disease = None
        response = ""
        if image_bytes:
            # Extract disease from image
            predicted_disease = self.extract_disease_from_image(image_bytes, device=device)
            print(predicted_disease)
            # Generate response using the disease-specific prompt
            response = self.llm.invoke(self.get_disease.format(
                user_input=user_input,
                conversation_history=conversation_history,
                predicted_disease=predicted_disease
            ))
        else:
            # No image provided; prompt the user to upload an image
            response = self.llm.invoke(self.get_main_prompt.format(
                user_input=user_input,
                conversation_history=conversation_history
            ))

        return {
                    'response': response.content,
                    'predicted_disease': ""
                }
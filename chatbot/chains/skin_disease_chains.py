# chatbot/chains/skin_disease_chains.py

from .base_chains import BaseChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from PIL import Image
import io
from actual_models.skin_disease_model import predict

from ..utils import query_refiner_models, guard_base_skin
logger = logging.getLogger(__name__)

class SkinDiseaseChain(BaseChain):
    def __init__(self, disease_model, llm, class_to_idx):
        """
        Initializes the SkinDiseaseChain.

        Args:
            disease_model (SkinDiseaseModel): The skin disease classification model.
            llm (ChatOpenAI): The language model.
            class_to_idx (dict): Mapping from class names to indices.
        """
        self.disease_model = disease_model
        self.llm = llm
        self.get_main_prompt = self.main_prompt()
        self.get_disease = self.return_disease()
        self.awaiting_image = False  # Flag to track if the chain is waiting for an image
        self.class_to_idx = class_to_idx
        self.image_provided = False  # Flag to track if an image has been provided and processed

    def main_prompt(self):
        """
        Creates the LLMChain with a specific prompt template.

        Returns:
            PromptTemplate: The initialized prompt template.
        """

        template = """
You are a friendly medical assistant specialized in diagnosing skin diseases based on images, encourage the user to give you and image of the problem in the skin.

Do not use the phrase "Skin Disease Doctor: or Nurse:, be normal"

f the User talks in arabic answer in english and tell him to switch to the Arabic feature on thew button on the top right.

User input: {user_input}

Conversation history: {conversation_history}

    """


        return PromptTemplate(
            input_variables=["user_input", "conversation_history", "image_uploaded"],
            template=template
        )

    def return_disease(self):
        """
        Creates a prompt template for providing information based on the predicted disease.

        Returns:
            PromptTemplate: The initialized prompt template with disease-specific instructions.
        """
        template = """
        You are a friendly Skin Disease Doctor. Inform the user about their disease results naturally without mentioning numbers directly. 
        This is the contact number 140 for the Lebanese Red Cross (for all health emergencies), if the disease is severe prompt the user for this emergency number.

        Predicted Disease: {predicted_disease}
        
        ### Instructions:
        - Use a natural and conversational tone.
        - Do not say phrases like "User said" or "According to your input."
        - Only use information from the prediction or direct user input.
        - Do not generate any additional insights or answers on your own.

        ### Constraints:
        - Do not mention or reveal any internal metrics or confidence scores.
        - Do not provide medical advice beyond the diagnosis.
        - Do not use technical jargon that the user might not understand.
        - Ensure clarity and empathy in your response.

        Conversation History:
        {conversation_history}

        User Input:
        {user_input}

        """

        return PromptTemplate(
            input_variables=["user_input", "conversation_history", "predicted_disease"],
            template=template
        )

    def generate_response(self, user_input: str, conversation_history: str, image_path: str = None, device: str = 'cpu') -> dict:
        """
        Generates a response based on user input, conversation history, and an optional image.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path (str, optional): Path to the image related to skin conditions, if provided.
            device (str, optional): Device to perform computation ('cpu' or 'cuda').

        Returns:
            dict: A dictionary containing the chatbot's response and any additional data.
        """
        
        
        predicted_disease = None
        response = ""

        if image_path:
                try:
                    # Process the image and predict the disease
                    predicted_disease = predict( self.disease_model,
                        image_path
                    )
                    logger.info(f"Predicted Disease: {predicted_disease}")

                    # Set the flag indicating that an image has been provided and diagnosed
                    self.image_provided = True

                    # Generate response using the disease-specific prompt
                    disease_prompt = self.get_disease.format(
                        user_input=user_input,
                        conversation_history=conversation_history,
                        predicted_disease=predicted_disease
                    )
                    response = self.llm.invoke(disease_prompt)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    response = "I'm sorry, but I couldn't process the image you provided. Could you please try uploading it again?"
        else:
                # No image provided; prompt the user to upload an image
                main_prompt = self.get_main_prompt.format(
                    user_input=user_input,
                    conversation_history=conversation_history,
                    image_uploaded=self.image_provided
                )
                response = self.llm.invoke(main_prompt)

            # Handle LLM response object
        if hasattr(response, 'content'):
                final_response = response.content
        else:
                final_response = response

        return {
                'response': final_response,
                'predicted_disease': predicted_disease if predicted_disease else ""
            }
    
        

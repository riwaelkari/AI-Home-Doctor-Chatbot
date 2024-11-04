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
You are a friendly medical assistant specialized in diagnosing skin diseases based on images.

User input: {user_input}

Conversation history: {conversation_history}


        ### Instructions:
        -Greet the user and say you are the skin disease doctor.
        -Ask the user gently to provide you with the picture of the possible skin disease
        - Respond only based on the image provided and the conversation history.
        - Be friendly and empathetic.
        - Avoid giving advice beyond what can be inferred from the image diagnosis.
        - Gently redirect the user to provide a new image only if necessary.
        - If the user asks for general advice or next steps, politely remind them that you're here to assist with image-based diagnoses and suggest they consult a healthcare professional for further guidance.

        ### Constraints:
        - Do **not** provide medical advice beyond the scope of image-based diagnosis.
        - Maintain a natural and conversational tone throughout the interaction.
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

        **Predicted Disease:** {predicted_disease}

        **Conversation History:**
        {conversation_history}

        **User Input:**
        {user_input}

        ### Instructions:
        - Use a natural and conversational tone.
        - Do not say phrases like "User said" or "According to your input."
        - Only use information from the prediction or direct user input.
        - Do not generate any additional insights or answers on your own.

        ### Constraints:
        - Do **not** mention or reveal any internal metrics or confidence scores.
        - Do **not** provide medical advice beyond the diagnosis.
        - Do **not** use technical jargon that the user might not understand.
        - Ensure clarity and empathy in your response.
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
                predicted_disease = self.disease_model.predict(
                    image_path,
                    device=device,
                    class_to_index=self.class_to_idx
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

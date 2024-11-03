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
            LLMChain: The initialized chain.
        """
        

        template = """
You are a friendly medical assistant specialized in diagnosing skin diseases based on images.

- If the user has already uploaded an image, do not ask for another one unless the user mentions they want to upload a new image. Continue with the conversation based on the image that was provided.
- If no image has been uploaded yet, kindly ask the user to upload a picture of their skin issue using the attach button on the left side of the text box.
- Be polite and empathetic, reminding the user that you can only diagnose based on the image they provide.
- If the user asks for general advice or next steps, politely remind them that you're here to assist with image-based diagnoses and suggest they consult a healthcare professional for further guidance if needed.

User input: {user_input}

Conversation history: {conversation_history}

- Respond only based on the image provided and the conversation history. Be friendly, avoid giving advice beyond what can be inferred from the image diagnosis, and gently redirect the user to provide a new image only if necessary.
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
You are a friendly home doctor. Inform the User with his disease results without mentioning numbers directly: {predicted_disease}
*Conversation History:* 
{conversation_history}

*User Input:* 
{user_input}


Be natural, dont say: User said
Only use information from the prediction or direct user input. Do not generate any additional insights or answers on your own.
*Do not mention
"""
        return PromptTemplate(
            input_variables=["user_input","conversation_history","predicted_disease"],template= template)


##########################################################################################################################


    def generate_response(self, user_input: str, conversation_history: str, image_path, device='cpu') -> dict:
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

      
        if image_path:
            # Extract disease from image
            predicted_disease = self.disease_model.predict(image_path, device=device,class_to_index = self.class_to_idx)
            print(predicted_disease)


            # Append 'image processed' to the conversation history for future checks
            #conversation_history += "\n[Image Processed]"
            # Set the flag indicating that an image has been provided and diagnosed
            self.image_provided = True
            # Generate response using the disease-specific prompt
            response = self.llm.invoke(self.get_disease.format(
                user_input=user_input,
                conversation_history=conversation_history,
                predicted_disease=predicted_disease
            ))
        #elif self.image_provided:
        #    # If an image was already provided and diagnosed, respond based on that diagnosis
        #    response = "I have already analyzed the provided image. If you have another skin issue, please upload a new image. Otherwise, I can only diagnose based on images."

          # Check if the conversation history already contains a processed image
        #elif "image processed" in conversation_history.lower():
        #    print("wosil lahon lezim")
        #    # Image has already been processed, respond accordingly
        #    response = "I have already analyzed the provided image. If you need further assistance, please upload a new image. Otherwise, I can only diagnose based on images."

        else:
            # No image provided; prompt the user to upload an image
            response = self.llm.invoke(self.get_main_prompt.format(
                user_input=user_input,
                conversation_history=conversation_history
            ))

            # Check if the response is from the LLM and has a `.content` attribute
        if hasattr(response, 'content'):
            # It's an LLM response with a content attribute
            final_response = response.content
        else:
            # It's a manually assigned string response
            final_response = response

        return {
                    'response': final_response,
                    'predicted_disease': ""
                }
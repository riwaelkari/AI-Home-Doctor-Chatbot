# server/skin_disease_setup.py

import os
import logging
from dotenv import load_dotenv
from actual_models.skin_disease_model import SkinDiseaseClassifier, load_checkpoint  # Ensure correct import path
from chatbot.chains.skin_disease_chains import SkinDiseaseChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

def initialize_skin_disease_chain(agent):
    """
    Initializes the SkinDiseaseChain and registers it with the agent.
    
    Args:
        agent (Agent): The chatbot agent to register the chain with.
    """
    # Load environment variables
    load_dotenv()
    
    # Configure Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    

    
    # Load the trained model using the provided load_checkpoint function
    try:
        model = load_checkpoint('saved_models/checkpoint.pth')
        logger.info(f"SkinDiseaseClassifier loaded successfully from checkpoint")
    except FileNotFoundError as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"An error occurred while loading the SkinDiseaseClassifier: {e}")
        return
    
    # Initialize OpenAI Embeddings
    openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure 'OPENAI_API_KEY' is set in .env
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        return
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Initialize Chat Model with LangChain
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o-mini",  # Replace with your desired model
        openai_api_key=openai_api_key
    )
    
    # Initialize SkinDiseaseChain
    skin_disease_chain = SkinDiseaseChain(
        disease_model=model,
        llm=llm
    )
    
    # Register the chain with the agent
    agent.register_chain('skin_disease', skin_disease_chain)
    logger.info("SkinDiseaseChain initialized and registered with the agent.")

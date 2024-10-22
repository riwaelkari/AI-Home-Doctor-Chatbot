# server/skin_disease_setup.py

import os
import logging
from dotenv import load_dotenv
from chatbot.chains.base_agent_chains import BaseModelChain
from langchain_openai import ChatOpenAI

def initialize_base_chain(agent):
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

    
    # Initialize OpenAI Embeddings
    openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure 'OPENAI_API_KEY' is set in .env
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        return
        
    # Initialize Chat Model with LangChain
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o-mini",  # Replace with your desired model
        openai_api_key=openai_api_key
    )
    
    # Initialize SkinDiseaseChain
    skin_disease_chain = BaseModelChain(
        llm=llm
    )
    
    # Register the chain with the agent
    agent.register_chain('base_model', skin_disease_chain)
    logger.info("SkinDiseaseChain initialized and registered with the agent.")

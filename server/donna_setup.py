import os
import logging
from dotenv import load_dotenv
from chatbot.chains.donna_secretary_chains import DonnaChain
from langchain_openai import ChatOpenAI

def initialize_donna_chain(agent):
    """
    Initializes the DonnaChain and registers it with the agent.

    Args:
        agent (Agent): The chatbot agent to register the chain with.
    """
    # Load environment variables
    load_dotenv()

    # Configure Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize Chat Model with LangChain
    openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure 'SECRET_TOKEN' is set in .env
    if not openai_api_key:
        logger.error("SECRET_TOKEN not found in environment variables.")
        return

    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o-mini",  # Replace with your desired model
        openai_api_key=openai_api_key
    )

    # Initialize DonaChain
    donna_chain = DonnaChain(
        llm=llm
    )

    # Register the chain with the agent
    agent.register_chain('donna', donna_chain)
    logger.info("DonaChain initialized and registered with the agent.")

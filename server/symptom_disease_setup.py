# server/symptom_disease_setup.py

import os
import logging
from dotenv import load_dotenv
from actual_models.symptom_disease_model import SymptomDiseaseModel
from chatbot.chains.chains import SymptomDiseaseChain
from chatbot.data_processing import (
    load_data,
    preprocess_data,
    create_documents_from_df,
    split_docs,
    create_faiss_index
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def initialize_symptom_disease_chain(agent):
    """
    Initializes the SymptomDiseaseChain and registers it with the agent.
    
    Args:
        agent (Agent): The chatbot agent to register the chain with.
    """
    # Load environment variables
    load_dotenv()
    
    # Configure Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load Data
    symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
    training_data_cleaned, testing_data_cleaned, classes, all_symptoms, le = preprocess_data(symptom_df, testing_symptoms_df)
    
    # Initialize OpenAI Embeddings
    openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure 'OPENAI_API_KEY' is set in .env
    # Initialize Chat Model with LangChain
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4o-mini", 
        openai_api_key=openai_api_key
    )
    
    # Initialize and load the disease prediction model
    model = SymptomDiseaseModel()
    model.set_additional_attributes(all_symptoms, training_data_cleaned['prognosis_encoded'])
    model.load_model()
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create FAISS index
    dataframes = [description_df, precaution_df, severity_df]
    documents = create_documents_from_df(dataframes, types=["description", "precaution", "severity"])
    split_documents = split_docs(documents)
    faiss_store = create_faiss_index(split_documents, embeddings_model)
    
    # Initialize SymptomDiseaseChain
    symptom_disease_chain = SymptomDiseaseChain(
        all_symptoms=all_symptoms,
        disease_model=model,
        openai_api_key=openai_api_key,
        faiss_store=faiss_store,
        faiss_index=faiss_store.index,
        embeddings_model=embeddings_model,
        split_docs=split_documents
    )
    
    # Register the chain with the agent
    agent.register_chain('symptom_disease', symptom_disease_chain)
    logger.info("SymptomDiseaseChain initialized and registered with the agent.")

# server.py

from flask import Flask, request, jsonify
import openai
import os
from train_models.neural_network import SymptomDiseaseModel
import logging
from flask_cors import CORS
from data_processing import load_data, preprocess_data, prepare_documents
from chains import SymptomDiseaseChain
from utils import encode_user_symptoms_fromgpt
import numpy as np
import pandas as pd

# Updated imports from langchain-community 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory  # Assuming memory is still in langchain

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)

# Prepare Documents for LangChain
documents = prepare_documents(description_df, precaution_df, severity_df, symptom_df)

# Initialize OpenAI Embeddings
openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure this is set in your environment variables
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create FAISS Vector Store from Documents
vector_store = FAISS.from_texts(texts=documents, embedding=embeddings)

# Initialize Chat Model with LangChain-Community
llm = ChatOpenAI(
    temperature=0.7, #yew sm3t shi? bekaa al aweye? hadii 5?????? 
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key
)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize Conversational Retrieval Chain with explicit output_key
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    return_source_documents=True,
    output_key="answer"  # Explicitly set the output_key to 'answer'
)

# Set OpenAI API Key for symptom extraction GPT
openai.api_key = os.getenv('SECRET_TOKEN')  # Ensure the API key is set in your environment variables

# Prepare Training Data
y_train = training_data_cleaned['prognosis_encoded']

# Initialize and load the model
model = SymptomDiseaseModel(y_train)
model.load_model('../models/saved_model.h5')

# Initialize SymptomDiseaseChain
symptom_disease_chain = SymptomDiseaseChain(
    all_symptoms=all_symptoms,
    disease_model=model,
    classes=classes,
    openai_api_key=openai_api_key
)

# Function to handle detailed queries with LangChain
def handle_detailed_query(disease, query):
    refined_query = f"What is the {query} of {disease}?"
    try:
        qa_result = qa_chain.invoke({"question": refined_query})  # Use invoke instead of __call__ or run
        answer = qa_result.get('answer', '')
        source_documents = qa_result.get('source_documents', [])
        return {
            "detail": answer,
            "sources": [doc.metadata for doc in source_documents]
        }
    except Exception as e:
        logger.error(f"Error during detailed query handling: {e}")
        return {
            "error": f"An error occurred while fetching details: {e}"
        }


# Route to Handle Chat
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            logger.warning("No message provided in the request.")
            return jsonify({'error': 'No message provided.'}), 400
        user_message = data['messages']
        logger.info(f"Received message: {user_message}")

        # Use SymptomDiseaseChain to extract symptoms and predict disease
        prediction_result = symptom_disease_chain.predict_disease(user_message)

        if "error" in prediction_result:
            return jsonify({
                'gpt_response': prediction_result["error"]
            }), 200

        predicted_disease = prediction_result["predicted_disease"]
        extracted_symptoms = prediction_result["extracted_symptoms"]

        # Inform the user of the diagnosis and prompt for more details
        diagnosis_message = (
            f"You have been diagnosed with **{predicted_disease}** based on the symptoms: {', '.join(extracted_symptoms)}. "
            "You can ask me about the 'description', 'precautions', or 'severity' of your condition."
        )
        logger.info(f"Diagnosis: {diagnosis_message}")

        return jsonify({
            'gpt_response': diagnosis_message,
            'predicted_disease': predicted_disease
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

# New Route to Handle Detailed Queries with LangChain
@app.route('/details', methods=['POST'])
def details():
    try:
        data = request.get_json()
        if not data or 'disease' not in data or 'query' not in data:
            logger.warning("Insufficient data provided for details.")
            return jsonify({'error': 'Insufficient data provided.'}), 400

        disease = data['disease']
        query = data['query'].lower()
        logger.info(f"Received details request for disease: {disease}, query: {query}")

        # Validate query type
        if query not in ['description', 'precautions', 'severity']:
            return jsonify({'error': 'Invalid query type. Please ask about "description", "precautions", or "severity".'}), 400

        # Handle the detailed query using LangChain
        detail_result = handle_detailed_query(disease, query)

        return jsonify({
            'detail': detail_result['detail'],
            'sources': detail_result['sources']
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in details route: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

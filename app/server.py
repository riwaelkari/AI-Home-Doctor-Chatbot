#server.py
from flask import Flask, request, jsonify
import os
from train_models.neural_network import SymptomDiseaseModel
import logging
from flask_cors import CORS
from data_processing import load_data, preprocess_data,create_faiss_index,create_documents_from_df,split_docs
from chains import SymptomDiseaseChain
from langchain_community.vectorstores import FAISS #needs to go away
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage,HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings 

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 

# Configure Logging#doode  langchain_community.vectorstores.FAISS) is tightly coupled with the usage of pickle for saving and loading metadata alongside the FAISS index.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df = load_data()
training_data_cleaned, testing_data_cleaned, classes, all_symptoms = preprocess_data(symptom_df, testing_symptoms_df)

# Initialize OpenAI Embeddings
openai_api_key = os.getenv('SECRET_TOKEN')  # Ensure this is set in your environment variables

# Initialize Chat Model with LangChain-Community
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",  # Ensure you use a valid model name
    openai_api_key=openai_api_key
)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)
X_train = training_data_cleaned.drop(columns=['prognosis', 'prognosis_encoded'])

dataframes = [description_df, precaution_df, severity_df]
y_train = training_data_cleaned['prognosis_encoded']

# Initialize and load the model
model = SymptomDiseaseModel()
 # Input layer 

faiss_index_name = "chatbot_index"
#dataframes = [description_df, precaution_df, severity_df]
#documents = create_documents_from_df(dataframes)
#split_documents = split_docs(documents)
# Create FAISS index using your function
#faiss_store = create_faiss_index(split_documents, index_name="chatbot_index")

model.load_model("models/saved_model.h5")
symptom_disease_chain = SymptomDiseaseChain(
    all_symptoms=all_symptoms,
    disease_model=model,
    classes=classes,
    openai_api_key=openai_api_key,
    #faiss_store = faiss_store
)#yea heard and felt
#awiyi???????
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            logger.warning("No message provided in the request.")
            return jsonify({'error': 'No message provided.'}), 400
        user_message = data['messages']
        logger.info(f"Received message: {user_message}")

        memory.chat_memory.add_user_message(user_message)

        # Retrieve conversation history
        memory_variables = memory.load_memory_variables({})

        conversation_history = memory_variables.get('conversation_history', [])

        # Prepare conversation history as a formatted string
        formatted_history = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in conversation_history])
        # Generate response using SymptomDiseaseChain
        response_message, predicted_disease = symptom_disease_chain.generate_response(user_message, formatted_history)
        # Add response to memory
        memory.chat_memory.add_ai_message(AIMessage(content=response_message))

        # Prepare the response payload
        response_payload = {
            'gpt_response': response_message
        }

        if predicted_disease:
            response_payload['predicted_disease'] = predicted_disease

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

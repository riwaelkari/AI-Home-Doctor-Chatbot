# server/server.py

from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from chatbot.agent import Agent
from langchain.schema import AIMessage, HumanMessage
from .symptom_disease_setup import initialize_symptom_disease_chain
from langchain.memory import ConversationBufferMemory

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Agent
agent = Agent()

# Initialize SymptomDiseaseChain and register it with the agent
initialize_symptom_disease_chain(agent)

# Optionally, set the default chain (using 'symptom_disease' as default)
agent.set_default_chain(agent.chains.get('symptom_disease'))

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            logger.warning("No message provided in the request.")
            return jsonify({'error': 'No message provided.'}), 400
        
        user_message = data['messages']
        logger.info(f"Received message: {user_message}")

        # Retrieve conversation history
        memory_variables = memory.load_memory_variables({})
        conversation_history = memory_variables.get('conversation_history', [])

        # Prepare conversation history as a formatted string
        formatted_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
            for msg in conversation_history
        ])

        # Delegate to the agent
        response_dict = agent.handle_request(user_message, formatted_history)

        # Add user and assistant messages to memory
        memory.chat_memory.add_user_message(user_message)
        print(response_dict)
        memory.chat_memory.add_ai_message(response_dict.get('response',''))

        # Prepare the response payload
        response_payload = {
            'gpt_response': response_dict.get('response')
        }

        if 'predicted_disease' in response_dict and response_dict['predicted_disease']:
            response_payload['predicted_disease'] = response_dict['predicted_disease']

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

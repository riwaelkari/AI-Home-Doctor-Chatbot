from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
from flask_cors import CORS
from chatbot.agent import Agent
from langchain.schema import AIMessage, HumanMessage
from .skin_disease_setup import initialize_skin_disease_chain
from .symptom_disease_setup import initialize_symptom_disease_chain
from .base_chain_setup import initialize_base_chain
from langchain.memory import ConversationBufferMemory
import os

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Agent
agent = Agent()

# Initialize Chains
initialize_skin_disease_chain(agent)
initialize_symptom_disease_chain(agent)
initialize_base_chain(agent)
agent.set_default_chain(agent.chains.get('base_model'))

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask to accept file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files (CSS, JS)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Chat route for POST request
@app.route('/chat', methods=['POST'])
def chat():
    try:
        image_path = None  # Initialize image_path to None
        # Access form data and files
        message = request.form.get('message')
        image = request.files.get('image')

        # Debugging: Log received message and image
        print(f"Received message: {message}")
        print(f"Received files: {request.files}")
        if image:
            print(f"Image filename: {image.filename}")
            # Save the image to the uploads directory
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            print(f"Image saved to {image_path}")
        else:
            print("No image provided")

        if not message and not image:
            return jsonify({'error': 'No message or image provided.'}), 400

        if message.strip().lower() == "reset":
            memory.clear()  # Clear the conversation memory
            formatted_history = ''
            agent.set_default_chain(agent.chains.get('base_model'))

        # Retrieve conversation history
        memory_variables = memory.load_memory_variables({})
        conversation_history = memory_variables.get('conversation_history', [])

        # Prepare conversation history as a formatted string
        formatted_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
            for msg in conversation_history
        ])
        print(image_path)
        # Delegate to the agent
        response_dict = agent.handle_request(message, formatted_history, image_path)

        # Add user and assistant messages to memory
        memory.chat_memory.add_user_message(message)
        memory.chat_memory.add_ai_message(response_dict.get('response',''))
        print(response_dict.get('bot_icon', 'images/nurse_icon.png'))
        print(response_dict['bot_icon'])
        # Prepare the response payload
        response_payload = {
            'gpt_response': response_dict.get('response'),
            'bot_name': response_dict.get('bot_name', 'Nurse'),
            'bot_icon': response_dict.get('bot_icon', 'images/nurse_icon.png')
        }
        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=False)

from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

from chatbot.agent import Agent
from langchain.schema import AIMessage, HumanMessage
from .skin_disease_setup import initialize_skin_disease_chain
from .symptom_disease_setup import initialize_symptom_disease_chain
from .base_chain_setup import initialize_base_chain
from langchain.memory import ConversationBufferMemory
from .donna_setup import initialize_donna_chain

# Import the SpeechToTextModel
from actual_models.audiototext import SpeechToTextModel

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Flask to accept file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit maximum file size (e.g., 10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 Megabytes

# Initialize Speech-to-Text Model with desired model size
speech_to_text_model = SpeechToTextModel(model_size="base")  # Options: "tiny", "base", "small", "medium", "large"

# Initialize Agent
agent = Agent()

# Initialize Chains with Exception Handling
try:
    initialize_skin_disease_chain(agent)
    initialize_symptom_disease_chain(agent)
    initialize_base_chain(agent)
    initialize_donna_chain(agent)
    agent.set_default_chain(agent.chains.get('base_model'))
except Exception as e:
    logger.error(f"Error initializing chains: {e}", exc_info=True)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

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
        # Initialize variables
        image_path = None
        audio_path = None
        message = None
        language = 'En'  # Default language

        # Check if the request is JSON (e.g., reset request)
        if request.is_json:
            data = request.get_json()
            if data.get('reset'):
                # Reset the agent and memory
                memory.clear()
                agent.set_default_chain(agent.chains.get('base_model'))
                agent.set_nurse_chain(agent.chains.get('base_model'))
                logger.info("Agent and memory have been reset.")

                # Return initial greeting
                response_payload = {
                    'gpt_response': "Hello! How can I assist you today?",
                    'bot_name': 'Nurse',
                    'bot_icon': 'images/nurse_icon.png'
                }
                return jsonify(response_payload), 200
            else:
                # Handle normal JSON message (if any)
                message = data.get('message')
                language = data.get('language', 'En')
        else:
            # Handle form data (e.g., message with image or audio)
            message = request.form.get('message')
            language = request.form.get('language', 'En')
            logger.info(f"Received language: {language}")

            image = request.files.get('image')
            audio = request.files.get('audio')

            # Handle image upload
            if image:
                image_filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image.save(image_path)
                logger.info(f"Image saved to {image_path}")

            # Handle audio upload
            if audio:
                audio_filename = secure_filename(audio.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                audio.save(audio_path)
                logger.info(f"Audio saved to {audio_path}")

                # Transcribe the audio using SpeechToTextModel
                try:
                    transcribed_text = speech_to_text_model.transcribe(audio_path)
                    logger.info(f"Transcribed Text: {transcribed_text}")
                    message = transcribed_text  # Use the transcribed text as the message
                except Exception as e:
                    logger.error(f"Error during transcription: {e}", exc_info=True)
                    return jsonify({'error': 'Audio transcription failed.'}), 500
        if not message and image and not audio:
            message = "Image provided"
        if not message and not image and not audio:
            return jsonify({'error': 'No message, image, or audio provided.'}), 400

        # Remove or comment out the old reset handling based on message content
        # if message and message.strip().lower() == "reset":
        #     memory.clear()
        #     # Implement any additional reset logic if necessary

        # Retrieve conversation history
        memory_variables = memory.load_memory_variables({})
        conversation_history = memory_variables.get('conversation_history', [])

        # Prepare conversation history as a formatted string
        if language == 'Ar':
            logger.info("Arabic language selected")
            # Translate the user's input from Arabic to English
            translated_message = agent.translate_text(message, 'English')
            logger.info(f"Translated user input to English: {translated_message}")
            user_input_en = translated_message
        else:
            logger.info("English language selected")
            user_input_en = message

        formatted_history = "\n".join([
            f"{'Patient' if isinstance(msg, HumanMessage) else msg.metadata.get('bot_name', 'Nurse')}: {msg.content}" 
            for msg in conversation_history
        ])

        # Delegate to the agent
        print(agent.current_chain)
        response_dict = agent.handle_request(user_input_en, formatted_history, image_path, language)

        # Add user and assistant messages to memory
        if message:
            memory.chat_memory.add_user_message(message)
        elif audio:
            memory.chat_memory.add_user_message("[Audio Message]")
        else:
            memory.chat_memory.add_user_message("[Image Uploaded]")

        ai_message = AIMessage(
            content=response_dict.get('response', ''),
            metadata={'bot_name': response_dict.get('bot_name', 'Nurse')}
        )

        memory.chat_memory.add_message(ai_message)

        if language == 'Ar':
            response = agent.translate_text(response_dict.get('response'), 'Arabic')
        else:
            response = response_dict.get('response')

        # Prepare the response payload
        response_payload = {
            'gpt_response': response,
            'bot_name': response_dict.get('bot_name', 'Nurse'),
            'bot_icon': response_dict.get('bot_icon', 'images/nurse_icon.png')
        }
        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host ="0.0.0.0",port=5000,debug=False)

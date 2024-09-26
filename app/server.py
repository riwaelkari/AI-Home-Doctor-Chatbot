from flask import Flask, request, jsonify
import openai
import os
import logging
from flask_cors import CORS  # Add this for CORS handling

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
openai.api_key = os.getenv('SECRET_TOKEN')  # Ensure the API key is set in your environment variables

def gpt_response(message):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",  # Change to your GPT model of choice
            messages=[
                {"role": "system", "content": "You are a home doctor chatbot."},
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in GPT response: {e}")
        return "I'm sorry, but I couldn't process your request at the moment."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("No message provided in the request.")
            return jsonify({'error': 'No message provided.'}), 400
        
        user_message = data['message']
        logger.info(f"Received message: {user_message}")
        
        # Get GPT response based on user's message
        gpt_reply = gpt_response(user_message)
        logger.info("GPT response generated successfully.")
        
        return jsonify({
            'gpt_response': gpt_reply
        }), 200
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

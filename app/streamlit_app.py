# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/predict"  # Ensure Flask server is running

st.set_page_config(page_title="Home Doctor Chatbot", page_icon="🩺")

st.title("🏥 Home Doctor Chatbot")

# Initialize session state to keep track of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! Are you feeling unwell today?"}
    ]
    st.session_state.awaiting_symptoms = False  # Flag to track if symptoms are expected

# Function to send user input and receive response
def send_message(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check if the app is awaiting symptoms input
    if st.session_state.get('awaiting_symptoms', False):
        # Assume the user is providing symptoms
        symptoms = [sym.strip() for sym in user_input.split(",")]
        try:
            response = requests.post(API_URL, json={"symptoms": symptoms})
            if response.status_code == 200:
                data = response.json()
                predicted_disease = data.get("predicted_disease", "Unknown")
                gpt_response = data.get("gpt_response", "No additional information.")
                st.session_state.messages.append({
                    "role": "bot", 
                    "content": f"**Predicted Disease:** {predicted_disease}\n\n**Doctor says:** {gpt_response}"
                })
            else:
                # Attempt to extract error message from response
                try:
                    data = response.json()
                    error_message = data.get("error", "An error occurred.")
                except:
                    error_message = "An error occurred."
                st.session_state.messages.append({"role": "bot", "content": f"Error: {error_message}"})
        except Exception as e:
            st.session_state.messages.append({"role": "bot", "content": f"Error: {str(e)}"})
        finally:
            st.session_state.awaiting_symptoms = False  # Reset the flag
    else:
        # Initial response handling
        if user_input.strip().lower() in ["yes", "y", "i am not feeling well", "yes, I am"]:
            st.session_state.messages.append({"role": "bot", "content": "I'm sorry to hear that. Please enter your symptoms separated by commas."})
            st.session_state.awaiting_symptoms = True  # Set the flag to await symptoms
        else:
            st.session_state.messages.append({"role": "bot", "content": "I'm here to help if you need anything."})

# Chat Interface
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Doctor:** {message['content']}")

# User input
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    send_message(user_input)
    # No need to call st.experimental_rerun()

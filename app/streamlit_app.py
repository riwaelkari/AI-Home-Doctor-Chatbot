# streamlit_app.py

import streamlit as st
import requests
from PIL import Image

API_CHAT_URL = "http://127.0.0.1:5000/chat"      # Ensure Flask server is running
# API_DETAILS_URL = "http://127.0.0.1:5000/details"  # Endpoint for details (if needed)

st.set_page_config(page_title="Home Doctor Chatbot", page_icon="🩺")
icon = Image.open("images/logo-1-removebg.png")

col1, col2 = st.columns([1, 8])  # Adjust the width ratio as needed (1:8)

# Display the logo in the first column
with col1:
    st.image(icon, width=95)  # Adjust the width as needed

# Display the title in the second column
with col2:
    st.title("Home Doctor Chatbot")

# Initialize session state to keep track of the conversation and disease
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! How can I assist you today?"}
    ]

if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None

# Function to send user input and receive response from GPT
def send_message(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        response = requests.post(API_CHAT_URL, json={"messages": user_input})
        if response.status_code == 200:
            data = response.json()
            gpt_response = data.get("gpt_response", "No response from the chatbot.")
            predicted_disease = data.get("predicted_disease", None)
            st.session_state.predicted_disease = predicted_disease
            st.session_state.messages.append({
                "role": "bot",
                "content": gpt_response
            })
        else:
            st.session_state.messages.append({"role": "bot", "content": "Error: Failed to get a response."})
       
    except Exception as e:
        st.session_state.messages.append({"role": "bot", "content": f"Error: {str(e)}"})

# User input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    send_message(user_input)

# Display conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Doctor:** {message['content']}")

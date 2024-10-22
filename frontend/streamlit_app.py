import streamlit as st
import requests
import base64
import time
from PIL import Image

API_CHAT_URL = "http://127.0.0.1:5000/chat"  # Ensure Flask server is running

favicon_path = "images/stethoscope.png"  # Path to your favicon file

# Convert the favicon to base64
with open(favicon_path, "rb") as f:
    favicon_base64 = base64.b64encode(f.read()).decode()

# Set the page configuration
st.set_page_config(
    page_title="Home Doctor Chatbot",
    layout="wide",
    page_icon=f"data:image/png;base64,{favicon_base64}"
)

# Convert logo and user images to base64 for embedding
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_base64_image("images/logo-1-removebg.png")
user_base64 = get_base64_image("images/user.png")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set overall background and text color */
    body {
        background-color: #0E1117;
        color: #f1f1f1;
    }

    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: #0E1117;
        display: flex;
        flex-direction: column;
        height: 100vh;
        justify-content: flex-start; /* Align content to the top */
        overflow: hidden; /* Prevents scrolling */
        width: 250px; /* Fixed width to control expansion */
        transition: width 0.3s ease; /* Smooth transition for sidebar */
    }

    /* Ensure the image fits within the sidebar without expanding */
    .sidebar .sidebar-content img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
        margin-left: 0; /* Align image to the left */
        margin-bottom: 10px; /* Space below the image */
    }

    /* Chat bubble styles */
    .chat-bubble {
        border-radius: 15px;
        padding: 8px;
        margin: 3px 0;
        max-width: 65%;
        display: inline-block;
        align-items: flex-start;
        position: relative;
    }

    .bot-bubble {
        background-color: #262730;
        color: #fff;
        text-align: left;
        margin-left: 5px;
    }

    .user-bubble {
        background-color: #831434;
        color: #fff;
        text-align: right;
        margin-right: 5px;
    }

    .chat-container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        margin-bottom: 5px;
    }

    .user-chat-container {
        display: flex;
        justify-content: flex-end; /* Align user messages to the right */
        width: 100%;
    }

    .user-chat-container .chat-content {
        margin-right: 10px; /* Space between message and icon */
    }

    /* Fixed input container at the bottom */
    .fixed-input-container {
        position: fixed;
        bottom: 0; /* Positioned at the very bottom */
        left: 0;
        width: 100%;
        background-color: #0E1117; /* Match the main background */
        padding: 10px 15px; /* Padding for input area */
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        z-index: 1000; /* Ensure it stays above other elements */
    }

    /* Input row (chat input and attach button) */
    .input-row {
        display: flex;
        width: 100%;
        max-width: 800px; /* Adjust as needed */
    }

    /* Decrease the height and font size of the input box */
    .fixed-input-container .stTextInput > div > div > input {
        height: 35px; /* Adjusted height of the input box */
        font-size: 14px; /* Adjusted font size for better fit */
    }

    /* Message icons */
    .message-icon {
        width: 50px;
        height: 40px;
        margin: 10px;
    }

    .chat-content {
        display: flex;
        flex-direction: column;
    }

    .sender-label {
        font-weight: bold;
        font-size: 0.9em;
        margin-bottom: 3px;
        line-height: 1;
    }

    /* Chat display area */
    .chat-display {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        overflow-y: auto;
        padding: 15px;
        margin: 0;
        height: calc(100vh - 250px);  /* Adjusted height to accommodate fixed input */
    }

    /* Hide the default Streamlit status */
    .stStatus {
        display: none;
    }

    /* Custom status text */
    .custom-status {
        position: fixed;
        top: 10px;
        right: 10px;
        color: #f1f1f1;
        font-weight: bold;
        font-size: 1em;
        z-index: 1001; /* Above other elements */
    }

    /* Disclaimer under the input box */
    .input-disclaimer {
        padding: 0; /* Remove existing padding */
        height: 15px; /* Approximately 5mm */
        background-color: #0E1117; /* Match the main background */
        color: #D3D3D3; /* Set text color to light gray */
        font-size: 0.8em; /* Adjusted font size */
        text-align: center;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        margin-top: 5px; /* Space above disclaimer */
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Optional: Adjust scrollbar appearance for better aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #444;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add custom status text "Diagnosing" in the top right corner
st.markdown("<div class='custom-status'>Diagnosing</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("images/logo-1-removebg.png")  # Image will respect the fixed width and CSS
st.sidebar.title("Home Doctor 🩺")  # Added stethoscope emoji
st.sidebar.markdown("Your personal health assistant.")

# Initialize session state to keep track of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! How can I assist you today?"}
    ]

# Initialize session state for the uploader visibility
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

def toggle_uploader():
    st.session_state.show_uploader = not st.session_state.show_uploader

# Callback function for sending messages
def send_message(user_input):
    # Append the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # **Added: Temporary "Diagnosing..." message**
    diagnosing_message = {"role": "bot", "content": "Diagnosing..."}
    st.session_state.messages.append(diagnosing_message)
    
    try:
        # Make the API call
        response = requests.post(API_CHAT_URL, json={"messages": user_input})
        if response.status_code == 200:
            data = response.json()
            gpt_response = data.get("gpt_response", "No response from the chatbot.")
            predicted_disease = data.get("predicted_disease", None)
            st.session_state.predicted_disease = predicted_disease
            
            # **Added: Replace "Diagnosing..." with actual response**
            st.session_state.messages.pop()  # Remove the diagnosing message
            st.session_state.messages.append({
                "role": "bot",
                "content": gpt_response
            })
        else:
            # **Added: Replace "Diagnosing..." with error message**
            st.session_state.messages.pop()
            st.session_state.messages.append({"role": "bot", "content": "Error: Failed to get a response."})
    except Exception as e:
        # **Added: Replace "Diagnosing..." with exception message**
        st.session_state.messages.pop()
        st.session_state.messages.append({"role": "bot", "content": f"Error: {str(e)}"})

# Handle file uploads
if st.session_state.show_uploader:
    st.markdown("<div style='padding: 15px;'>", unsafe_allow_html=True)
    with st.container():
        file = st.file_uploader("Upload your data")
        if file:
            with st.spinner("Processing your file"):
                time.sleep(5)  # Dummy wait for demo purposes
                # You can add your file processing logic here
                st.success("File uploaded and processed successfully!")
                # Optionally, append the file information to the chat
                st.session_state.messages.append({
                    "role": "bot",
                    "content": f"Received your file: {file.name}"
                })
    st.markdown("</div>", unsafe_allow_html=True)

# Display conversation with icons
st.markdown("<div class='chat-display'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class='chat-container user-chat-container'>
                <div class='chat-content user-bubble chat-bubble'>
                    <span class='sender-label'>You:</span> <span>{message['content']}</span>
                </div>
                <img src='data:image/png;base64,{user_base64}' class='message-icon'/>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='chat-container'>
                <img src='data:image/png;base64,{logo_base64}' class='message-icon'/>
                <div class='chat-content bot-bubble chat-bubble'>
                    <span class='sender-label'>Doctor:</span> <span>{message['content']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# User input and Attach button within the fixed input container
st.markdown("<div class='fixed-input-container'>", unsafe_allow_html=True)

# Create a container for the input row (chat input and attach button)
st.markdown("<div class='input-row'>", unsafe_allow_html=True)
input_col, attach_col = st.columns([4, 1])

with input_col:
    user_input = st.chat_input("You:")

with attach_col:
    attach_button = st.button("📎 Attach", key="attach_button", on_click=toggle_uploader)

st.markdown("</div>", unsafe_allow_html=True)  # Close input-row div

# Add the disclaimer within the fixed input container
st.markdown(
    """
    <div class='input-disclaimer'>
        <p>AI Home Doctor can make mistakes. Please check back with a professional.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)  # Close fixed-input-container div

if user_input:
    send_message(user_input)

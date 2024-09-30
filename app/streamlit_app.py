import streamlit as st
import requests
import base64

API_CHAT_URL = "http://127.0.0.1:5000/chat"  # Ensure Flask server is running

st.set_page_config(page_title="Home Doctor Chatbot", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #f1f1f1;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
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
        background-color: #444;
        color: #fff;
        text-align: left;
        margin-left: 5px;
    }
    .user-bubble {
        background-color: #831434;
        color: #fff;
        text-align: right;
        margin-left: auto;
        margin-right: 5px;
    }
    .chat-container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        margin-bottom: 5px;
    }
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        padding: 15px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }
    .message-icon {
        width: 40px;
        height: 40px;
        margin-right: 10px;
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
    .chat-display {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        overflow-y: auto;
        padding: 0;
        margin: 0;
        height: calc(100vh - 200px);  /* Adjusted height to reduce the gap */
    }
    .chat-form-container {
        display: flex;
        align-items: center;
        margin-top: 10px;
    }
    .text-input {
        flex-grow: 1;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("images/logo-1-removebg.png", use_column_width=True)
st.sidebar.title("Home Doctor")
st.sidebar.markdown("Your personal health assistant.")

# Convert logo to base64 for embedding
logo_path = "images/logo-1-removebg.png"
with open(logo_path, "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()
user_icon_path = "images/user.png"
with open(user_icon_path, "rb") as image_file:
    user_base64 = base64.b64encode(image_file.read()).decode()


# Initialize session state to keep track of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! How can I assist you today?"}
    ]

# Callback function for sending messages
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

# User input using st.chat_input
user_input = st.chat_input("You:")

if user_input:
    send_message(user_input)

# Display conversation with icon for the doctor
st.markdown("<div class='chat-display'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class='chat-container'>
                <img src='data:image/png;base64,{user_base64}' class='message-icon'/>
                <div class='chat-content user-bubble chat-bubble'>
                    <span class='sender-label'>You:</span> <span>{message['content']}</span>
                </div>
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
        #ok
st.markdown("</div>", unsafe_allow_html=True)
#Let nounou cook
#nounou is burnt
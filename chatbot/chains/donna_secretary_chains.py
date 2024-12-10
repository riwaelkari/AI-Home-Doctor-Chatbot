import re
import dateparser
from dotenv import load_dotenv  
from .base_chains import BaseChain
from langchain.prompts import PromptTemplate
import logging
import threading
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time
import json
from dateutil.parser import parse as parse_date
import os
import uuid
from ..utils import query_refiner_models,guard_donna
# Configure logging
logger = logging.getLogger(__name__)

class DonnaChain(BaseChain):
    _instance = None  # <-- Singleton instance
    _lock = threading.Lock()  # <-- Lock for thread-safe instantiation


    def __init__(self, llm):
        """
        Initializes the DonnaChain.
        Args:
            llm (ChatOpenAI): The language model.
        """
        # Prevent re-initialization of the singleton instance
        if hasattr(self, '_initialized') and self._initialized:
            return

        load_dotenv()  # <-- Load environment variables from .env file

        self.llm = llm
        self.get_prompt = self.main_prompt()
        self.prescriptions = []
        self.email_thread = threading.Thread(target=self.send_reminder_emails)
        self.email_thread.daemon = True
        self.email_thread.start()
        logger.info("DonnaChain initialized and email reminder thread started.")
        self._initialized = True  # <-- Mark as initialized


    def main_prompt(self):
        """
        Creates the prompt template for interacting with the user.
        Returns:
            PromptTemplate: The initialized prompt template.
        """
        template = """
You are Donna, a friendly and efficient medical secretary. You help users schedule their medication reminders.

Instructions:
- Greet the user and offer assistance in setting up medication reminders and saying bye.
- Ask the user to collect from him the medication name, dosage, timing, and user's email address.
- Confirm the schedule with the user.
- Thank the user and inform them that you will send reminders accordingly.
_do not reply with "Donna:" 
- Do not provide medical advice.
- Ensure privacy by not sharing any personal information elsewhere.

Conversation:
{conversation}

"""
        return PromptTemplate(
            input_variables=["conversation"],
            template=template
        )

    def generate_response(self, user_input: str, conversation: str, image_path: str = None) -> dict:
        """
        Generates a response based on user input and conversation history.
        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path (str, optional): Not used in DonnaChain.
        Returns:
            dict: A dictionary containing the chatbot's response and any additional data.
        """
        #guard_response = guard_base_donna
        guard_response = guard_donna(user_input) #apply guard rails here
        if (guard_response == 'allowed'):
            # Generate response to the user
            prompt = self.get_prompt.format(
                conversation=conversation
            )
            response = self.llm.invoke(prompt)

            # Handle LLM response object
            final_response = response.content if hasattr(response, 'content') else response

            # Enhanced Extraction Prompt to Enforce JSON within Code Blocks
            extraction_prompt = f"""
    Please extract the following information from the user's input and conversation history:
    - Medication name
    - Dosage
    - Timing (including time of day and frequency)
    - User's email address
    - Greet the user and offer assistance in setting up medication reminders and saying bye.
- Ask the user to collect from him the medication name, dosage, timing, and user's email address.
- Confirm the schedule with the user.
- Thank the user and inform them that you will send reminders accordingly.
- Do not reply with "Donna: or Nurse:" 
- Do not provide medical advice.
- Something issue or something relating to the reminders or they failed or some issue with reminders or emails
- Ensure privacy by not sharing any personal information elsewhere.


    Provide the information in strict JSON format with the keys: medication, dosage, timing, email.

    Enclose the JSON in triple backticks and specify 'json' for syntax highlighting.

    Example:

    ```json
    {{
        "medication": "Aspirin",
        "dosage": "1 pill",
        "timing": "in 5 minutes",
        "email": "user@example.com"
    }}
    ```
    Ensure that the JSON is the only content in your response. Do not include any additional text.

    Conversation:
    {conversation}
    """

            extraction_response = self.llm.invoke(extraction_prompt)

            # Enhanced JSON Parsing to Handle Code Blocks and Additional Text
            try:
                raw_content = extraction_response.content.strip()

                # Attempt to extract JSON from code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)

                if json_match:
                    json_content = json_match.group(1)
                    logger.debug("Extracted JSON from code block.")
                else:
                    # Assume the entire content is JSON
                    json_content = raw_content
                    logger.debug("Assuming entire content is JSON.")

                prescription_data = json.loads(json_content)

                # Validate that all required keys are present
                required_keys = {"medication", "dosage", "timing", "email"}
                if not required_keys.issubset(prescription_data.keys()):
                    missing = required_keys - prescription_data.keys()
                    raise ValueError(f"Missing keys in JSON: {missing}")

                # Parse timing to get the next reminder time using dateparser
                next_reminder = dateparser.parse(prescription_data['timing'], settings={'RELATIVE_BASE': datetime.now()})
                frequency_delta = (next_reminder - datetime.now()).total_seconds()
                if not next_reminder:
                    raise ValueError("Unable to parse timing.")
                if frequency_delta <= 0:#new
                         raise ValueError("Timing must be in the future.")#new

                prescription_data['next_reminder'] = next_reminder
                prescription_data['frequency_seconds'] = frequency_delta #new

                # Assign a unique ID
                prescription_data['id'] = str(uuid.uuid4())

                # Check for duplicate prescriptions
                if not any(
                    p['email'] == prescription_data['email'] and
                    p['medication'].lower() == prescription_data['medication'].lower() and
                    p['dosage'].lower() == prescription_data['dosage'].lower() and
                    p['timing'] == prescription_data['timing']
                    for p in self.prescriptions
                ):
                    # Add to prescriptions list
                    self.prescriptions.append(prescription_data)
                    logger.info(f"Added prescription: {prescription_data}")
                else:
                    logger.warning(f"Duplicate prescription detected: {prescription_data}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}")
                logger.debug(f"Extraction response content: {extraction_response.content}")
            except Exception as e:
                logger.error(f"Failed to extract prescription data: {e}")

                if 'extraction_response' in locals():
                    logger.debug(f"Extraction response content: {extraction_response.content}")
                else:
                    logger.debug("extraction_response is not defined.")

            return {
                'response': final_response
            }
        else:
            return {"response": guard_response}


    def send_reminder_emails(self):
        """
        Background thread function to send reminder emails at the scheduled times.
        """
        logger.info("Email reminder thread started.")
        while True:
            now = datetime.now()
            logger.debug(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            for prescription in self.prescriptions:
                logger.debug(f"Evaluating prescription: {prescription}")
                if 'next_reminder' in prescription and prescription['next_reminder'] <= now:
                    logger.info(f"Sending reminder email to {prescription['email']}")
                    # Send email
                    self.send_email(
                        to_email=prescription['email'],
                        subject="Medication Reminder: " + prescription['medication'],
                        message=f"Reminder: Please take {prescription['dosage']} of {prescription['medication']}."
                    )
                    # Schedule next reminder (assuming daily for simplicity)
                    prescription['next_reminder'] += timedelta(days=1)
                    logger.info(f"Sent reminder email to {prescription['email']}")
            time.sleep(60)  # Check every minute

    def send_email(self, to_email, subject, message):
        """
        Sends an email using SMTP.
        Args:
            to_email (str): Recipient email address.
            subject (str): Email subject.
            message (str): Email body.
        """

        # Configure your SMTP server settings
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        smtp_username = os.getenv('GMAIL_USER')
        smtp_password = os.getenv('GMAIL_PASS')

        # Add logging to verify which SMTP user is being used
        logger.debug(f"SMTP Username (GMAIL_USER): {smtp_username}")

        # Create the email message
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = smtp_username
        msg['To'] = to_email

        # Send the email via SMTP
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(smtp_username, [to_email], msg.as_string())
            server.quit()
            logger.info(f"Email sent to {to_email}")
        except smtplib.SMTPAuthenticationError:
            logger.error("Failed to authenticate with the SMTP server. Check your username and password.")
        except smtplib.SMTPConnectError:
            logger.error("Failed to connect to the SMTP server. Check your network connection.")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ method to implement Singleton Pattern.
        Ensures only one instance of DonnaChain exists.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DonnaChain, cls).__new__(cls)
        return cls._instance
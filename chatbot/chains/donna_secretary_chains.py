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
logger = logging.getLogger(__name__)

class DonnaChain(BaseChain):
    def __init__(self, llm):
        """
        Initializes the DonnaChain.

        Args:
            llm (ChatOpenAI): The language model.
        """
        self.llm = llm
        self.get_prompt = self.main_prompt()
        self.prescriptions = []
        self.email_thread = threading.Thread(target=self.send_reminder_emails)
        self.email_thread.daemon = True
        self.email_thread.start()

    def main_prompt(self):
        """
        Creates the prompt template for interacting with the user.

        Returns:
            PromptTemplate: The initialized prompt template.
        """
        template = """
You are Donna, a friendly and efficient medical secretary. You help users schedule their medication reminders.

Conversation history:
{conversation_history}

User input:
{user_input}

### Instructions:
- Greet the user and offer assistance in setting up medication reminders.
- Collect the medication name, dosage, timing, and user's email address.
- Confirm the schedule with the user.
- Thank the user and inform them that you will send reminders accordingly.

### Constraints:
- Use a natural and conversational tone.
- Do not provide medical advice.
- Ensure privacy by not sharing any personal information elsewhere.
"""
        return PromptTemplate(
            input_variables=["user_input", "conversation_history"],
            template=template
        )

    def generate_response(self, user_input: str, conversation_history: str, image_path: str = None) -> dict:
        """
        Generates a response based on user input and conversation history.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.
            image_path (str, optional): Not used in DonnaChain.

        Returns:
            dict: A dictionary containing the chatbot's response and any additional data.
        """
        # Generate response to the user
        prompt = self.get_prompt.format(
            user_input=user_input,
            conversation_history=conversation_history
        )
        response = self.llm.invoke(prompt)

        # Handle LLM response object
        final_response = response.content if hasattr(response, 'content') else response

        # Use the LLM to extract prescription details
        extraction_prompt = f"""
Please extract the following information from the user's input and conversation history:
- Medication name
- Dosage
- Timing (including time of day and frequency)
- User's email address

Provide the information in JSON format with the keys: medication, dosage, timing, email.

Conversation history:
{conversation_history}

User input:
{user_input}
"""
        extraction_response = self.llm.invoke(extraction_prompt)

        # Parse the JSON output
        try:
            prescription_data = json.loads(extraction_response.content)
            # Parse timing to get the next reminder time
            next_reminder = parse_date(prescription_data['timing'])
            prescription_data['next_reminder'] = next_reminder
            # Add to prescriptions list
            self.prescriptions.append(prescription_data)
            logger.info(f"Added prescription: {prescription_data}")
        except Exception as e:
            logger.error(f"Failed to extract prescription data: {e}")

        return {
            'response': final_response
        }

    def send_reminder_emails(self):
        """
        Background thread function to send reminder emails at the scheduled times.
        """
        while True:
            now = datetime.now()
            for prescription in self.prescriptions:
                if 'next_reminder' in prescription and prescription['next_reminder'] <= now:
                    # Send email
                    self.send_email(
                        to_email=prescription['email'],
                        subject="Medication Reminder",
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
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")

from utils import  query_refiner_severity

# Test the function
conversation_log = """
Assistant: It seems like you may have symptoms like fever, cough, and headache.
User: What can I do now?
"""
user_query = "What should I be worried about?"

refined_questions = query_refiner_severity(conversation_log, user_query)
print(refined_questions)

# Expected output: ["What is the severity of fever?", "What is the severity of cough?", "What is the severity of headache?"]

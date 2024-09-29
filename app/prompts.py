# prompts.py

from langchain.prompts import PromptTemplate

symptom_extraction_prompt = PromptTemplate(
    input_variables=["user_input", "symptom_list"],
    template="""
You are a medical assistant. Extract the symptoms from the following user input based on the provided list of possible symptoms.

Possible symptoms: {symptom_list}

User input: {user_input}

Extracted symptoms (as a comma-separated list):
"""
)

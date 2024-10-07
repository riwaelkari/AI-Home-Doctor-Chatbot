# chains.py
from data_processing import get_similar_docs
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils import encode_user_symptoms_fromgpt,query_refiner, find_match,query_refiner_severity
from data_processing import get_similar_docs,get_diseases_by_symptoms
from data_processing import calc_severity_of_disease
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger(__name__)
class SymptomDiseaseChain:
    def __init__(self, all_symptoms, disease_model, openai_api_key, faiss_store, faiss_index, embeddings_model, split_docs):
        """
        Initializes the SymptomDiseaseChain with necessary components.

        Args:
            all_symptoms (list): List of all possible symptoms. 
            disease_model (object): Trained disease prediction model.
            classes (list): List of disease classes.
            openai_api_key (str): OpenAI API key.
        """



        self.all_symptoms = all_symptoms
        self.disease_model = disease_model
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model="gpt-4o-mini"
        )
        self.split_docs = split_docs
        self.main_prompt = self.get_main_prompt()
        self.symptom_extraction_prompt = self.get_symptom_extraction_prompt()
        self.disease_prompt = self.get_disease_prompt()
        self.get_info_prompt = self.return_info_prompt()
        self.get_severity_prompt=self.return_severity_prompt()
        self.faiss_store = faiss_store  # Add FAISS store as an attribute
        self.faiss_index = faiss_index
        self.embeddings_model = embeddings_model

        self.current_disease = None
        self.current_symptoms = []




    def get_main_prompt(self):
        """
        Defines the prompt template for symptom extraction.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly medical assistant at home, interact with the user {user_input} and mainly Ask him to give you his symptoms.

Conversation history: {conversation_history}

Answer based on Conversation history also.
"""

        return PromptTemplate(
            input_variables=["user_input","conversation_history"],
            template=template
        )       



    def get_symptom_extraction_prompt(self):
            """
            Defines the prompt template for symptom extraction.

            Returns:
                PromptTemplate: The formatted prompt template.
            """
            template = """
    You are a specialized medical assistant focused on extracting symptoms from the user's input. Follow these instructions strictly:

    Inputs Provided:
    - Possible symptoms: {symptom_list}
    - User input: {user_input}
    - Conversation history: {conversation_history}
    *DO DIRECTLY: If there are no symptoms in User input ALONE DIRECTLY RETURN NOTHING AND DO NOT READ THE FOLLOWING STEPS.*
    Extraction Rules:
    1. Do *not* extract symptoms from the conversation history unless there are symptoms in User inpurt
    2. *If no symptoms are found in the current user input, return the exact phrase: NOTHING*.

    Output Format:
    - If symptoms are found IN USER INPUT: Return a comma-separated list of symptoms including the ones in conversation histroy (e.g., "fever, cough").
    - If no symptoms are found: Return "NOTHING" without any additional text.

    Ensure that you *strictly follow these instructions* without exceptions. Any deviation will lead to incorrect outcomes.
     IMPORTANT: **please only check conversation history for additional symptoms if symptoms are provided in User input and include them in the return.
     *PLEASE DO NOT CHECK CONVERSATION HISTORY IF there are no symptoms in User input and return the phrase: NOTHING*
    """

            return PromptTemplate(
                input_variables=["user_input", "symptom_list","conversation_history"],
                template=template
            )



    def get_disease_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """

    You are a friendly and empathetic home doctor. Based on the conversation history, you should notify the user that they might have {disease}. Then tell them they could ask about the Description and precautions of the disease, and the severity of their symptoms.

    Number of diseases that include the user's symptoms: {n}

    If the number of diseases that include the user's symptoms is greater than 1, tell the user to give more symptoms to not get misdiagnosed otherwise do not mention the problem at all. DO NOT EXPLICITLY MENTION THE NUMBER OF MATCHING DISEASES

    **Be reasonable  with the number i gave you and answer with importance based on the number**, meaning if the number is 1 dont mention to the user anything, if its 2 or 3 tell him he might get misdiagnosed, if its higher increase the caution of the message

    Conversation history: {conversation_history}

    User input: {user_input}

    Do not give the user additional info about the diseases.

    Response:
    """
        return PromptTemplate(
            input_variables=["disease", "n", "conversation_history", "user_input"],
            template=template
        )


    def return_info_prompt(self):
        template = """
You are a friendly and empathetic home doctor. Based on the conversation history and the currently diagnosed disease "{disease}", you should provide the following information exactly as it is provided below.

*Provided Information:*
{info}

*Conversation History:* 
{conversation_history}

*User Input:* 
{user_input}

*Instructions:*
1. Analyze the user's input to determine if the question is about the "description" or "precautions" of the disease.
2. 
    - If the user is asking for a *description*, provide a clear and concise description of the disease using the provided information.
    - If the user is asking for *precautions, list the **four (4)* most relevant precautions as (Dotted) bullet points under each other.
3. *Do not* include any information that is not present in the *Provided Information* section.
4. Ensure that your response is well-formatted, clear, and directly addresses the user's query.

*Response:*
"""
        return PromptTemplate(
            input_variables=["info", "conversation_history", "user_input", "disease"],
            template=template
        )

        
    def return_severity_prompt(self):
        template = """
        You are a friendly and empathetic home doctor. Based on the conversation history and the currently diagnosed disease {disease}, you should provide the following severity information exactly as it is provided below.

        Severity Level: {real_severity}

        If you think the severity information is not relevant to the user's question based on the conversation history, inform them accordingly.

        *Use only the provided severity information below in your answer and be brief:*

        *Conversation history:*
        {conversation_history}

        *User input:*
        {user_input}

        """
        return PromptTemplate(
            input_variables=["conversation_history", "user_input", "disease", "real_severity"],
            template=template
        )
################################################################################


    def extract_symptoms(self, user_input,conversation_history):
        """
        Extracts symptoms from the user input using the LLM.

        Args:
            user_input (str): The user's input message.

        Returns:
            list: A list of extracted symptoms.
        """#
        prompt_text = self.symptom_extraction_prompt.format(
            user_input=user_input,
            symptom_list=', '.join(self.all_symptoms),
            conversation_history = conversation_history
            )
        print(user_input)
        try:
            response = self.llm.invoke(prompt_text)
            logger.info(f"LLM Response for Symptom Extraction: {response}")
        except Exception as e:
            logger.error(f"Error while getting LLM response for symptom extraction: {e}")
            return []

        # Extract text from response
        if hasattr(response, 'content'):
            # For AIMessage objects
            text = response.content
        elif isinstance(response, dict):
            # For dictionary responses, if any
            text = response.get('content', '')
        elif isinstance(response, str):
            # For string responses
            text = response
        else:
            # Unexpected response type
            logger.error(f"Unexpected response type from LLM: {type(response)}")
            return []

        logger.info(f"Extracted Text: {text}")

        if text.strip().lower() == "no symptoms detected.":
            return []
        print(text)
        # Parse the response into a list of symptoms
        symptoms = [
        symptom.strip().lower()  # Normalize the symptoms

        for symptom in text.split(',')  # Split by comma to get individual symptoms
        ]
        print(symptoms)
        logger.info(f"Extracted Symptoms: {symptoms}")
        valid_symptoms = [symptom for symptom in symptoms if symptom in self.all_symptoms]
        return valid_symptoms

    def predict_disease(self, symptoms):
        print("here")
        if not isinstance(symptoms, list):
            symptoms = [symptoms]
        print(symptoms)
        current_disease = self.disease_model.predict_disease(symptoms)
        print(current_disease[0][0])
        self.current_symptoms = symptoms
        self.current_disease = current_disease[0][0]
        return current_disease[0][0]

######################################################################################


    def generate_response(self, user_input, conv_history):
        """
        Generates a response based on user input by extracting symptoms and predicting disease.

        Args:
            user_input (str): The user's input message.
            conversation_history (str): The history of the conversation.

        Returns:
            str: The chatbot's response.
            str or None: The predicted disease if available.
        """
        # Extract symptoms from user input
        symptoms = self.extract_symptoms(user_input,conv_history)
        print("symptoms are (if any)", symptoms)
        refined_query = query_refiner(user_input,self.current_disease)
        print(refined_query)
        if symptoms:
            # Predict disease based on extracted symptoms
            predicted_disease = self.predict_disease(symptoms)
                # Update conversation history with disease and symptoms
            #n = matching_disease_fre(symptoms,....)
            print(type(conv_history))
            n=get_diseases_by_symptoms(symptoms)
            print(n)
            response_message = self.llm.invoke(self.disease_prompt.format(
                disease=predicted_disease,
                n=n,
                conversation_history=conv_history,
                user_input=user_input
            ))

            
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
        elif any(keyword in refined_query for keyword in ["description", "precautions"]):
            print("entered right location")
            if "description" in refined_query.lower():
                desired_section = "description"
            elif "precautions" in refined_query.lower():
                desired_section = "precaution"
            similar_docs = get_similar_docs(refined_query, self.embeddings_model, self.faiss_index, self.split_docs, k=1,desired_type=desired_section)
            if similar_docs:
                info = similar_docs[0][0].page_content
            else:
                info = "No information available regarding your query."
            print(info)
            response_message = self.llm.invoke(self.get_info_prompt.format(
                    info = info,
                    conversation_history=conv_history,
                    user_input=user_input,
                    disease=self.current_disease  # Pass the current disease
            ))
            predicted_disease = None



        elif "severity" in refined_query.lower():
            print("wrong loc bad")
            # Use the updated query_refiner_severity function to generate severity-related questions
            refined_severity_queries = query_refiner_severity(conv_history, user_input)
            severity_responses = []
            list_severity = []
            info_number=0

            for severity_query in refined_severity_queries:
                # Use get_similar_docs to retrieve information about severity
                similar_docs = get_similar_docs(severity_query, self.embeddings_model, self.faiss_index, self.split_docs, k=1, desired_type="severity")
                if similar_docs:
                    info_severity = similar_docs[0][0].page_content
                    info_number=info_severity[-1]
                    
                else:
                    info_severity = "No information available regarding the severity of this symptom."
                    info_number = 0
                print(info_number)
                severity_responses.append(f"Question: {severity_query} Answer: {info_severity}")
                list_severity.append(int(info_number))
                print(list_severity)
            real_severity="" 
            real_severity = calc_severity_of_disease(list_severity)
            # Combine all severity responses into a single response message
            response_message1 = "\n".join(severity_responses)
            print(response_message1)
            response_message = self.llm.invoke(self.get_severity_prompt.format(
                    conversation_history=conv_history,
                    user_input=user_input,
                    disease=self.current_disease,
                    real_severity=real_severity  # Pass the current disease
            ))
            predicted_disease = None

        else:
            print("wrong loc 2")
            # No symptoms or asking about info is detected, generate a prompt to ask for symptoms
            response_message = self.llm.invoke(self.main_prompt.format(
                user_input=user_input,
                conversation_history = conv_history
            )) 
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            predicted_disease = None
        return response_message.content, predicted_disease
    




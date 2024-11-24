# chains.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from ..utils import query_refiner, query_refiner_severity
from actual_models.symptom_data_processing import get_similar_docs,get_diseases_by_symptoms,calc_severity_of_disease
import numpy as np
import logging
from .base_chains import BaseChain
# Configure Logging
logger = logging.getLogger(__name__)
class SymptomDiseaseChain(BaseChain):
    def __init__(self, all_symptoms, disease_model, openai_api_key, faiss_store, faiss_index, embeddings_model, split_docs,llm):
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
        self.llm = llm
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
You are a friendly and empathetic symptom disease doctor at home that encourages the user to give you their symptoms.


- Needed Information for Conversation:
- You are a doctor that has the ability to take user symptoms and give the users back their potential disease
- You can also give the description and precautions of the disease and the severity of the symptoms
- Do not mention any information about the disease, even descriptions and precautions and severity, but tell the Patient they can request them

Do not use the phrase "Symptom Disease Doctor:"
If the User talks in arabic answer in english and tell him to switch to the Arabic feature on thew button on the top right.

Conversation History:
{conversation_history}

User Input:
{user_input}

Response:
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
You are a medical assistant specializing in extracting symptoms from user input. Follow these instructions strictly:

- Possible Symptoms: {symptom_list}
- User Input: {user_input}

Instructions:

1. Symptom Extraction:
   - Examine the User Input and identify any symptoms that match the Possible Symptoms list.
   - Only consider the User Input; do not consider any conversation history.
   - If symptoms are found, return them as a comma-separated list in lowercase.
   - If no symptoms are found, return `NOTHING`.

Output Format:

- If symptoms are found:
  - fever, cough
- If no symptoms are found:
  - `NOTHING`

Important:

- Do not provide any explanations or additional text.
- Do not include any symptoms from the conversation history.
- Strictly adhere to the instructions.

Examples:

- User Input: "I've been having headaches and nausea."
  - Output: headache, nausea
- User Input: "Can you tell me more about the precautions for flu?"
  - Output: NOTHING
    """
        return PromptTemplate(template=template, input_variables=["symptom_list", "user_input", "conversation_history"])




    def get_disease_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly and empathetic home doctor.

Based on the user's symptoms, inform them that they might have {disease}. Let them know they can ask about the description and precautions of the disease, as well as the severity of their symptoms.
This is the contact number 140 for the Lebanese Red Cross (for all health emergencies), if the disease is severe prompt the user for this emergency number, based on his symptoms.
Instructions:

- If the number of matching diseases ({n}) is greater than 1, encourage the user to provide more symptoms to avoid misdiagnosis.
- Do not explicitly mention the number of matching diseases.
- Be reasonable with your concern based on the number provided:
  - If n is 1, proceed without mentioning any issues.
  - If n is 2 or 3, gently suggest that more symptoms could help in an accurate diagnosis.
  - If n is higher, express concern and emphasize the importance of sharing additional symptoms, only sharing additional symptoms.

Constraints:

- Do not provide additional information about the diseases at this point.
- Avoid medical jargon; keep the language simple and clear.
- Do not mention the exact number of matching diseases.

Conversation History:
{conversation_history}

User Input:
{user_input}


"""
        return PromptTemplate(
            input_variables=["disease", "n", "conversation_history", "user_input"],
            template=template
        )


    def return_info_prompt(self):
        template = """
You are a friendly and empathetic home doctor.

Instructions:

1. Determine if the user is asking about the "description" or "precautions" of the disease.
2. If asking for a description, provide a clear and concise description using the provided information.
3. If asking for precautions, list the four most relevant precautions as bullet points.

Constraints:

- Use only the provided information; do not add any additional details.
- Present the information clearly and professionally.
- Do not modify the provided information.

Currently Diagnosed Disease:
"{disease}"

Provided Information:
{info}

Conversation History:
{conversation_history}

User Input:
{user_input}
"""
        return PromptTemplate(
            input_variables=["info", "conversation_history", "user_input", "disease"],
            template=template
        )

        
    def return_severity_prompt(self):
        template = """
You are a friendly and empathetic home doctor.

Instructions:

- If the severity information is relevant to the user's question, provide it briefly and clearly.
- If not relevant, politely inform the user that the severity information may not address their concern.

Constraints:

- Use only the provided severity information in your response.
- Be concise and avoid unnecessary details.
- Do not provide additional medical advice unless asked.

Currently Diagnosed Disease:
"{disease}"

Severity Level:
{real_severity}

Conversation History:
{conversation_history}

User Input:
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


    def generate_response(self, user_input, conv_history, image_path):
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
        logger.info(f"Extracted symptoms: {symptoms}")
        refined_query = query_refiner(user_input,self.current_disease)
        logger.info(f"Refined query: {refined_query}")
        if image_path:
            response = "This Model does not support images, please either choose a model that does, or refrain from attaching images."
            return {"response": response,
                }
        elif symptoms:
            # Predict disease based on extracted symptoms
            predicted_disease = self.predict_disease(symptoms)
            logger.info(f"Predicted disease: {predicted_disease}")
                # Update conversation history with disease and symptoms
            #n = matching_disease_fre(symptoms,....)
            n=get_diseases_by_symptoms(symptoms)
            logger.info(f"Number of matching diseases: {n}")
            print(predicted_disease)
            response_message = self.llm.invoke(self.disease_prompt.format(
                disease=predicted_disease,
                n=n,
                conversation_history=conv_history,
                user_input=user_input
            ))

            print(response_message)
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            return {
                'response': response_message.content,
                'predicted_disease': predicted_disease
            }
        elif any(keyword in refined_query for keyword in ["description", "precautions"]):
            logger.info("User is requesting description or precautions.")
            if "description" in refined_query.lower():
                desired_section = "description"
            elif "precautions" in refined_query.lower():
                desired_section = "precaution"
            similar_docs = get_similar_docs(refined_query, self.embeddings_model, self.faiss_index, self.split_docs, k=1,desired_type=desired_section)
            if similar_docs:
                info = similar_docs[0][0].page_content
            else:
                info = "No information available regarding your query."
            logger.info(f"Retrieved info: {info}")
            response_message = self.llm.invoke(self.get_info_prompt.format(
                    info = info,
                    conversation_history=conv_history,
                    user_input=user_input,
                    disease=self.current_disease  # Pass the current disease
            ))
            predicted_disease = None
            return {
                'response': response_message.content,
                'predicted_disease': ""
            }



        elif "severity" in refined_query.lower():
            logger.info("User is requesting severity information.")
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
            return {
                'response': response_message.content,
                'predicted_disease': ""
            }

        else:
            logger.info("No symptoms or specific queries detected. Prompting for symptoms.")
            # No symptoms or asking about info is detected, generate a prompt to ask for symptoms
            response_message = self.llm.invoke(self.main_prompt.format(
                user_input=user_input,
                conversation_history = conv_history
            )) 
            logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
            predicted_disease = None
            return {
                'response': response_message.content,
            }    


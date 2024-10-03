# chains.py
from data_processing import get_similar_docs
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils import encode_user_symptoms_fromgpt,query_refiner, find_match,query_refiner_severity
from data_processing import get_similar_docs
import numpy as np
import logging

# Configure Logging
logger = logging.getLogger(__name__)
class SymptomDiseaseChain:
    def __init__(self, all_symptoms, disease_model, classes, openai_api_key, faiss_store, faiss_index, embeddings_model, split_docs):
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
        self.classes = classes
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
        self.faiss_store = faiss_store  # Add FAISS store as an attribute
        self.faiss_index = faiss_index
        self.embeddings_model = embeddings_model

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
You are a specialized medical assistant focused solely on extracting symptoms from the user's input. Adhere strictly to the following instructions:

**Inputs Provided:**
- **Possible symptoms (Confidential):** {symptom_list}
- **User input:** {user_input}
- **Conversation history:** {conversation_history}

**Extraction Rules:**

1. **Primary Analysis:**
   - **Check for Symptoms in User Input:**
     - **If the `user_input` contains one or more symptoms** from the `Possible symptoms` list:
       - **Extract all matching symptoms** from the `user_input`.
       - **Additionally**, extract any relevant symptoms from the `conversation_history`.
       - **Return** a **comma-separated list** of all extracted symptoms **only**.
     - **If the `user_input` does NOT contain any symptoms:**
       - **Ignore the `conversation_history` entirely**. Do not extract or consider any information from it.
       - **Return NOTHING**.

2. **Handling Non-Symptom Queries:**
   - **If the `user_input` is a question or statement unrelated to symptoms** (e.g., asking for descriptions, precautions, severity, greetings):
     - **Do NOT extract or return any symptoms**, even if they are present in the `conversation_history`.
     - **Return NOTHING**.

**Important Guidelines:**
- **Exclusivity:** The extraction should **only** be based on the `user_input` unless the `user_input` explicitly contains symptoms.
- **No Inference:** Do **not** infer or extract symptoms from the `conversation_history` if the `user_input` lacks symptoms.
- **Output Format:** **Only** return a **comma-separated list** of symptoms or **NOTHING**. **Do not** include any additional explanations, messages, or text.
- **Case Sensitivity:** Treat symptom matching as **case-insensitive**.
- **No Partial Matches:** Only extract complete symptom terms as listed in the `Possible symptoms`.

**Example Scenarios:**

- **Scenario 1:**
  - *User input:* "I've been experiencing a skin rash and itching."
  - *Bot response:* "skin_rash, itching"

- **Scenario 2:**
  - *User input:* "Can you describe the disease?"
  - *Bot response:* ""

- **Scenario 3:**
  - *User input:* "Hello!"
  - *Bot response:* ""

- **Scenario 4:**
  - *User input:* "I have a headache."
  - *Conversation history:* "User: I've been experiencing a skin rash and itching."
  - *Bot response:* "headache, skin_rash, itching"

- **Scenario 5:**
  - *User input:* "Please provide precautions."
  - *Conversation history:* "User: I've been experiencing a skin rash and itching."
  - *Bot response:* ""

**Additional Notes:**
- **Strict Adherence:** Ensure that these rules are followed **strictly** without exception.
- **Avoid Ambiguity:** The model should **never** consider `conversation_history` unless the `user_input` explicitly contains symptoms.
- **Testing:** After implementing this prompt, perform extensive testing with various user inputs to ensure consistent behavior.

By implementing these stricter and more explicit instructions, the model should consistently ignore the `conversation_history` when the `user_input` doesn't contain any symptoms, regardless of previous interactions.
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
You are a friendly and empathetic home doctor. Based on the conversation history, you should notify the user with the following disease {disease} and then tell him he could ask about the Description and precautions of the disease, and the severity of his symptoms.

Conversation history: {conversation_history}

User input: {user_input}

Do not give the user additional info about the disease.

Response:
"""
        return PromptTemplate(
            input_variables=["disease","conversation_history", "user_input"],
            template=template
        )
    



    def return_info_prompt(self):
        """
        Defines the prompt template for response generation.

        Returns:
            PromptTemplate: The formatted prompt template.
        """
        template = """
You are a friendly and empathetic home doctor. Based on the conversation history, you should give this info to the user asked for based on his symtoms and his disease (found in conversation history)

The info you should give: {info} if you think the info is not relevant to his question based on conversation history, tell him

Conversation history: {conversation_history}

User input: {user_input}

Only use the info you are given.

Response:
"""
        return PromptTemplate(
            input_variables=["info", "conversation_history", "user_input"],
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
        """
        Predicts the disease based on the extracted symptoms.

        Args:
            symptoms (list): List of extracted symptoms.

        Returns:
            dict: A dictionary containing the predicted disease and extracted symptoms.
        """
        # Step 1: Encode Symptoms and Predict Disease
        X_input = encode_user_symptoms_fromgpt(symptoms, self.all_symptoms)
        prediction = self.disease_model.predict(X_input)
        predicted_disease = self.classes[np.argmax(prediction)]

        logger.info(f"Predicted Disease: {predicted_disease}")

        return {
            "predicted_disease": predicted_disease,
            "extracted_symptoms": symptoms
        }


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
        refined_query = query_refiner(conv_history,user_input)
        print("these are the symptoms gpt has extracted if any", symptoms)
        print(refined_query)
        if symptoms:
            # Predict disease based on extracted symptoms
            prediction_result = self.predict_disease(symptoms)
            print("wrong loc")
            if "error" in prediction_result:
                response_message = prediction_result["error"]
                predicted_disease = None
            else:
                predicted_disease = prediction_result["predicted_disease"]
                print(type(conv_history))
                response_message = self.llm.invoke(self.disease_prompt.format(
                    disease = predicted_disease,
                    conversation_history=conv_history,
                    user_input=user_input
                ))

                logger.info(f"Diagnosis and GPT-Generated Response: {response_message}")
        elif any(keyword in refined_query for keyword in ["description", "precautions"]):
            print("entered right location")
            similar_docs = get_similar_docs(refined_query, self.embeddings_model, self.faiss_index, self.split_docs, k=2)
            if similar_docs:
                info = similar_docs[0][0].page_content
            else:
                info = "No information available regarding your query."
            print(info)
            response_message = self.llm.invoke(self.get_info_prompt.format(
                    info = info,
                    conversation_history=conv_history,
                    user_input=user_input
            ))
            predicted_disease = None



        elif "severity" in refined_query.lower():
            # Use the updated query_refiner_severity function to generate severity-related questions
            refined_severity_queries = query_refiner_severity(conv_history, user_input)
            severity_responses = []

            for severity_query in refined_severity_queries:
                # Use get_similar_docs to retrieve information about severity
                similar_docs = get_similar_docs(severity_query, self.embeddings_model, self.faiss_index, self.split_docs, k=1)
                if similar_docs:
                    info_severity = similar_docs[0][0].page_content
                else:
                    info_severity = "No information available regarding the severity of this symptom."
                severity_responses.append(f"Question: {severity_query} Answer: {info_severity}")

            # Combine all severity responses into a single response message
            response_message = "\n".join(severity_responses)
            print(response_message)
            predicted_disease = None
 #       elif "severity" in refined_query.lower():
 #           refined_severity_query=query_refiner_severity(conv_history,user_input)
  #          print("i have entered the severity part")
   #         # If the user is asking about the severity of symptoms
    #        print(refined_severity_query)
            
     #       severity_responses = []
      #      for symptomquestion in refined_severity_query:
       #         print("i am here 1")
                # Use get_similar_docs to retrieve information about severity
         #       similar_docs = get_similar_docs(symptomquestion, self.embeddings_model, self.faiss_index, self.split_docs, k=1)

           #     if similar_docs:
   #                 info_severity = similar_docs[0][0].page_content
      #              print("this is info severity",info_severity)
      #          else:
       #             info_severity = "No information available regarding the severity of this symptom."

 #               severity_responses.append(f"Question {symptomquestion}: .Answer: {info_severity}")

#                print(severity_responses)

        #    # Combine all severity responses into a single response message
        #    response_message1 = "\n".join(severity_responses)
        #    print(response_message1)
       #     response_message = self.llm.invoke(self.get_info_prompt.format(
       #             info = response_message1,
      #              conversation_history=conv_history,
       #             user_input=user_input
       #     ))
       #     predicted_disease = None


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
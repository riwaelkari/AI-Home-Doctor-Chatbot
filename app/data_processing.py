import pandas as pd
from sklearn.preprocessing import LabelEncoder
import faiss
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAIEmbeddings

def load_data():
    # Adjust the paths based on your directory structure
    symptom_df = pd.read_csv('../dataset/disease_symptoms_train.csv')
    description_df = pd.read_csv('../dataset/symptom_Description.csv')
    precaution_df = pd.read_csv('../dataset/symptom_precaution.csv')
    severity_df = pd.read_csv('../dataset/Symptom-severity.csv')
    testing_symptoms_df = pd.read_csv('../dataset/disease_symptoms_test.csv')
    return symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df

def preprocess_data(symptom_df, testing_symptoms):
    label_encoder = LabelEncoder()
    training_data_cleaned = symptom_df.copy()  # Use copy to avoid SettingWithCopyWarning
    training_data_cleaned['prognosis_encoded'] = label_encoder.fit_transform(training_data_cleaned['prognosis'])
    
    testing_data_cleaned = testing_symptoms.copy()
    testing_data_cleaned['prognosis_encoded'] = label_encoder.fit_transform(testing_data_cleaned['prognosis'])
    
    classes = label_encoder.classes_.tolist()
    all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure','swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
    'red_sore_around_nose', 'yellow_crust_ooze','fluid_overload'
    ]
    
    return training_data_cleaned, testing_data_cleaned, classes, all_symptoms

###Cast the loaded dox into a string 
def create_documents_from_df(dataframes, types):
    """
    Create Document objects from dataframes with metadata indicating the type.

    Args:
        dataframes (list): List of pandas DataFrames.
        types (list): List of strings indicating the type of each DataFrame.

    Returns:
        list: List of Document objects with metadata.
    """
    documents = []
    for df, doc_type in zip(dataframes, types):
        for _, row in df.iterrows():
            content = " ".join(row.astype(str).tolist())  # Combine row data into a string
            documents.append(Document(page_content=content, metadata={"type": doc_type}))
    return documents

###splitting 
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

###create and store embeddings
def create_embeddings(docs, embeddings_model):
    documents_text = [doc.page_content for doc in docs]
    embeddings_list = embeddings_model.embed_documents(documents_text)
    return np.array(embeddings_list)



def store_embeddings(embeddings, index_name="chatbot_index"):
    # Initialize a FAISS index
    dim = embeddings.shape[1]  # The dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # Use L2 distance for similarity
    index.add(embeddings)  # Add the embeddings to the index
    faiss.write_index(index, f"{index_name}.index")  # Save the index to disk
    return index  # Return the index for later use

def get_similar_docs(query, embeddings_model, index, split_documents, k, desired_type=None):
    """
    Retrieve similar documents based on the query and optionally filter by type.

    Args:
        query (str): The search query.
        embeddings_model: The embeddings model.
        index: The FAISS index.
        split_documents (list): List of Document objects.
        k (int): Number of similar documents to retrieve.
        desired_type (str, optional): The type of documents to filter by (e.g., 'precaution').

    Returns:
        list: List of tuples containing Document objects and their distances.
    """
    query_embedding = embeddings_model.embed_query(query)  # Embed the query
    distances, indices = index.search(np.array([query_embedding]), k * 2)  # Retrieve more to account for filtering

    similar_docs = []
    for j, i in enumerate(indices[0]):
        doc = split_documents[i]
        if desired_type:
            if doc.metadata.get("type") == desired_type:
                similar_docs.append((doc, distances[0][j]))
                if len(similar_docs) == k:
                    break
        else:
            similar_docs.append((doc, distances[0][j]))
            if len(similar_docs) == k:
                break

    return similar_docs


def create_faiss_index(docs, embeddings):
    """
    Creates FAISS index from documents and saves it.
    
    Args:
        docs (list): List of Document objects to be indexed.
        index_name (str): Name of the FAISS index file.
        
    Returns:
        FAISS: LangChain FAISS vector store.
    """    
    # Use FAISS with LangChain's wrapper
    faiss_store = FAISS.from_documents(docs, embeddings)
    
    # Save FAISS index for later use
    faiss_store.save_local("chatbot_index")
    return faiss_store

    
# Calculate the possible range of severity scores based on the symptom severity dataset
def calc_severity_of_disease(list_of_symtpoms_severities):
    severity_df = pd.read_csv('../dataset/Symptom-severity.csv')
    min_severity_weight = severity_df['weight'].min() #1
    max_severity_weight = severity_df['weight'].max() #7
    average_severity = sum(list_of_symtpoms_severities)/len(list_of_symtpoms_severities)
    # Define new threshold ranges based on the values from the dataset
    # Here, I use the range of severity weights to divide the categories more logically
   # def classify_severity_customized(average_severity):
    if average_severity < min_severity_weight + 1:
            return "Not Severe"
    elif min_severity_weight + 1 <= average_severity < min_severity_weight + 3:
            return "Moderate"
    elif min_severity_weight + 3 <= average_severity < max_severity_weight:
            return "Severe"
    else:
            return "Extremely Severe"

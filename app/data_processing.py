import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    # Adjust the paths based on your directory structure
    symptom_df = pd.read_csv('../dataset/disease_symptoms_train.csv')
    description_df = pd.read_csv('../dataset/symptom_Description.csv')
    precaution_df = pd.read_csv('../dataset/symptom_precaution.csv')
    severity_df = pd.read_csv('../dataset/Symptom-severity.csv')
    testing_symptoms_df= pd.read_csv('../dataset/disease_symptoms_test.csv')
    return symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df

def preprocess_data(symptom_df,testing_symptoms):
    training_data_cleaned = symptom_df
    label_encoder = LabelEncoder()
    training_data_cleaned['prognosis_encoded'] = label_encoder.fit_transform(training_data_cleaned['prognosis'])
    testing_data_cleaned = testing_symptoms.copy()
    testing_data_cleaned['prognosis_encoded'] = label_encoder.transform(testing_data_cleaned['prognosis'])
    classes = label_encoder.classes_.tolist()
    return training_data_cleaned, testing_data_cleaned, classes

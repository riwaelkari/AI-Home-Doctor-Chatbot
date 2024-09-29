import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_data():
    # Adjust the paths based on your directory structure
    symptom_df = pd.read_csv('../dataset/disease_symptoms_train.csv')
    description_df = pd.read_csv('../dataset/symptom_Description.csv')
    precaution_df = pd.read_csv('../dataset/symptom_precaution.csv')
    severity_df = pd.read_csv('../dataset/Symptom-severity.csv')
    testing_symptoms_df = pd.read_csv('../dataset/disease_symptoms_test.csv')
    return symptom_df, description_df, precaution_df, severity_df, testing_symptoms_df

def preprocess_data(symptom_df,testing_symptoms):
    training_data_cleaned = symptom_df
    label_encoder = LabelEncoder()
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
    return training_data_cleaned, testing_data_cleaned, classes ,all_symptoms

def prepare_documents(description_df, precaution_df, severity_df, symptom_df):
    """
    Prepares documents by merging disease descriptions, precautions, and computing disease severity.

    Args:
        description_df (DataFrame): Disease descriptions.
        precaution_df (DataFrame): Disease precautions.
        severity_df (DataFrame): Symptom severity scores.
        symptom_df (DataFrame): Disease to symptom mapping in one-hot encoding.

    Returns:
        list: A list of formatted document strings for each disease.
    """
    # Step 1: Standardize Column Names
    # Rename 'Description' to 'description' for consistency
    description_df.rename(columns={'Description': 'description'}, inplace=True)
    
    # Rename precaution columns to lowercase for consistency
    precaution_df.rename(columns={
        'Precaution_1': 'precaution_1',
        'Precaution_2': 'precaution_2',
        'Precaution_3': 'precaution_3',
        'Precaution_4': 'precaution_4'
    }, inplace=True)
    
    # Rename 'Symptom' to 'symptom' for consistency
    severity_df.rename(columns={'Symptom': 'symptom'}, inplace=True)
    
    # Step 2: Clean `severity_df` by removing erroneous rows
    # Remove any rows where 'symptom' is 'prognosis'
    severity_df = severity_df[severity_df['symptom'].str.lower() != 'prognosis']
    
    # Step 3: Melt `symptom_df` from wide to long format
    # Assuming 'symptom_df' has 'prognosis' and multiple symptom columns
    symptom_melted = symptom_df.melt(id_vars=['prognosis'], var_name='symptom', value_name='present')
    
    # Rename 'prognosis' to 'Disease' to match other DataFrames
    symptom_melted.rename(columns={'prognosis': 'Disease'}, inplace=True)
    
    # Step 4: Filter only present symptoms (where 'present' == 1)
    disease_symptom = symptom_melted[symptom_melted['present'] == 1].copy()
    
    # Step 5: Merge with `severity_df` to get 'weight' for each symptom
    disease_symptom_severity = disease_symptom.merge(severity_df, on='symptom', how='left')
    
    # Step 6: Handle missing severity weights by assigning a default value of 0
    disease_symptom_severity['weight'] = disease_symptom_severity['weight'].fillna(0)
    
    # Step 7: Compute total severity per disease by summing symptom weights
    disease_severity = disease_symptom_severity.groupby('Disease')['weight'].sum().reset_index()
    disease_severity.rename(columns={'weight': 'severity'}, inplace=True)
    
    # Step 8: Merge `description_df` and `precaution_df` on 'Disease'
    merged_df = description_df.merge(precaution_df, on='Disease', how='left')
    
    # Step 9: Merge computed `disease_severity` into `merged_df`
    merged_df = merged_df.merge(disease_severity, on='Disease', how='left')
    
    # Step 10: Handle any missing severity values by assigning a default value of 0
    merged_df['severity'] = merged_df['severity'].fillna(0)
    
    # Step 11: Prepare documents for each disease
    documents = []
    for _, row in merged_df.iterrows():
        disease = row['Disease']
        description = row['description']
        precautions = row['precaution_1']
        
        # Append additional precautions if they exist and are not null
        for i in range(2, 5):  # Assuming there are up to 4 precautions
            precaution_col = f'precaution_{i}'
            if precaution_col in row and pd.notnull(row[precaution_col]):
                precautions += f', {row[precaution_col]}'
        
        severity = row['severity']
        
        # Format the document string
        content = (
            f"Disease: {disease}\n"
            f"Description: {description}\n"
            f"Precautions: {precautions}\n"
            f"Severity: {severity}"
        )
        metadata = {"Disease": disease}
        
        documents.append(content)
    
    return documents
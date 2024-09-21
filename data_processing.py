import pandas as pd

# Load the symptom-disease dataset
symptom_df = pd.read_csv('dataset/dataset.csv')

# Load the disease description dataset
description_df = pd.read_csv('dataset/symptom_Description.csv')

# Load the disease precaution dataset
precaution_df = pd.read_csv('dataset/symptom_precaution.csv')

severity_df = pd.read_csv('dataset/Symptom-severity.csv')

# Display the first few rows to verify
print(symptom_df.head())
print(description_df.head())
print(precaution_df.head())
print(severity_df.head())
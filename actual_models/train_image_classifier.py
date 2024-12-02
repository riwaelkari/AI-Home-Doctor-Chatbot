# train.py

from skin_disease_model import (
    SkinDiseaseClassifier,
    get_input_args,
    load_data,
    train_model,
    save_checkpoint,
    load_checkpoint
)

import torch
from torch import nn, optim
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
def setup_dataset(data_dir):
    """
    Set up the dataset directory and load the class mapping for skin diseases.

    This function checks whether the specified dataset directory exists and raises an 
    error if not. It also defines a mapping of class indices to skin disease names, 
    which will be used during model training and evaluation.

    Args:
        data_dir (str): The directory where the dataset is stored.
 Raises:
        FileNotFoundError: If the dataset directory does not exist.

    Returns:
        dict: A dictionary mapping class indices (strings) to skin disease names.
    """
    # Define dataset directory
    MYDATA_DIR = Path(data_dir)

    # Check if the directory exists
    if not MYDATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory '{MYDATA_DIR}' not found. Please ensure the data directory exists.")
    else:
        print(f"[INFO] Dataset directory found at {MYDATA_DIR}")

    # Define the class mapping for the 23 skin diseases
    skin_diseases = {
    "0": "Warts Molluscum and other Viral Infections",
    "1": "Vasculitis Photos",
    "2": "Vascular Tumors",
    "3": "Urticaria Hives",
    "4": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "5": "Systemic Disease",
    "6": "Seborrheic Keratoses and other Benign Tumors",
    "7": "Scabies Lyme Disease and other Infestations and Bites",
    "8": "Psoriasis pictures Lichen Planus and related diseases",
    "9": "Poison Ivy Photos and other Contact Dermatitis",
    "10": "Nail Fungus and other Nail Disease",
    "11": "Melanoma Skin Cancer Nevi and Moles",
    "12": "Lupus and other Connective Tissue diseases",
    "13": "Light Diseases and Disorders of Pigmentation",
    "14": "Herpes HPV and other STDs Photos",
    "15": "Hair Loss Photos Alopecia and other Hair Diseases",
    "16": "Exanthems and Drug Eruptions",
    "17": "Eczema Photos",
    "18": "Cellulitis Impetigo and other Bacterial Infections",
    "19": "Bullous Disease Photos",
    "20": "Atopic Dermatitis Photos",
    "21": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "22": "Acne and Rosacea Photos"
}

    # Write the dictionary to a JSON file
    if not os.path.isfile('skin_disease_class_to_name.json'):
        with open('skin_disease_class_to_name.json', 'w') as file:
            json.dump(skin_diseases, file)
        print("[INFO] 'skin_disease_class_to_name.json' file has been created.")
    else:
        print("[INFO] 'skin_disease_class_to_name.json' file already exists.")

def main():
    args = get_input_args()

    if args.command == 'train':
        # Setup dataset
        setup_dataset(args.data_dir)

        # Training Mode
        # Load data
        trainloader, class_to_idx = load_data(args.data_dir)

        # Load the JSON mapping for skin disease class names
        with open('skin_disease_class_to_name.json', 'r') as file:
            class_mapping = json.load(file)
        print(f"[INFO] Loaded class mapping with {len(class_mapping)} classes.")

        # Build model
        model = SkinDiseaseClassifier(
            arch=args.arch,
            layer_1_hidden_units=args.layer_1_hidden_units,
            layer_2_hidden_units=args.layer_2_hidden_units,
            output_classes=len(class_mapping)
        )

        # Assign class_to_idx to model
        model.class_to_idx = class_to_idx

        # Define criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate
        )

        # Set device to GPU if available and requested
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training on device: {device}")

        # Train model
        model = train_model(model, trainloader, criterion, optimizer, args.epochs, device)

        # Save checkpoint
        save_checkpoint(
            model, args.save_dir, class_to_idx,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )

    elif args.command == 'predict':
        # Prediction Mode
        # Load the model
        model = load_checkpoint(args.checkpoint)
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

        # Run prediction
        probs, classes = model.predict(args.input, top_k=args.top_k, device=device)

        # Load category names if provided
        if args.category_names:
            try:
                with open(args.category_names, 'r') as f:
                    cat_to_name = json.load(f)
                classes = [cat_to_name.get(c, "Unknown") for c in classes]  # Handle missing categories
            except FileNotFoundError:
                print(f"Category names file {args.category_names} not found.")

        # Print results
        for i in range(len(classes)):
            print(f"Class: {classes[i]}, Probability: {probs[i]:.3f}")
    else:
        print("Please specify either 'train' or 'predict' as the command.")

if __name__ == "__main__":
    main()

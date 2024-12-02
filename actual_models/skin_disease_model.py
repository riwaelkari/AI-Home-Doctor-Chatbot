# train.py

import os
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
import numpy as np
import json
from torch.utils.data import random_split, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# ===========================
# Section 1: Model Definition
# ===========================

class SkinDiseaseClassifier(nn.Module):
    """
    A PyTorch model for classifying skin diseases based on images. This model
    utilizes a pre-trained ResNet18 architecture and fine-tunes it for the task
    of skin disease classification. It includes custom modifications to the 
    fully connected layer to adapt to the number of output classes for skin diseases.
    
    Args:
        arch (str): The architecture to use (e.g., 'resnet18'). Default is 'resnet18'.
        output_classes (int): The number of output classes (i.e., number of skin disease categories). Default is 23.
    """
    def __init__(self, arch='resnet18', output_classes=23):
        super(SkinDiseaseClassifier, self).__init__()

        # Load a pre-trained model based on the architecture selected
        if arch == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            input_features = self.model.fc.in_features # Get the input features to the fully connected layer
        else:
            raise ValueError(f"Architecture '{arch}' is not supported. Choose 'resnet18'.")

        # Freeze all pre-trained model parameters
        # Freezing means that these layers' weights will not be updated during training
        for param in self.model.parameters():
            param.requires_grad = False
#Freezing these layers ensures that you don't destroy the knowledge already captured by the pre-trained model, and it allows you to focus on fine-tuning the layers that are more task-specific.
#Freezing the majority of the model's layers reduces the number of parameters to train. This leads to faster training since only the newly added layers (e.g., your custom classifier) are updated during training.
        # Unfreeze layer3, layer4, and fc for fine-tuning
        # These layers will be updated during training
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True

        # Define an expanded classifier with two hidden layers, LayerNorm, and Dropout
        # Replace the final fully connected layer with a custom classifier
        # It includes two hidden layers, LayerNorm for normalization, and Dropout for regularization
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_features, 128)), # First fully connected layer with 128 units
            ('ln1', nn.LayerNorm(128)),  # Layer normalization after the first fully connected layer
            ('relu1', nn.ReLU()),# ReLU activation function
            ('dropout1', nn.Dropout(0.5)), # Dropout with a probability of 0.5 to reduce overfitting

            ('fc2', nn.Linear(128, 64)),
            ('ln2', nn.LayerNorm(64)),  
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),

            ('fc3', nn.Linear(64, output_classes)) # Final output layer with the number of output classes
        ]))

        # Save model architecture details
        self.arch = arch
        self.output_classes = output_classes

    def forward(self, x):
        """
        Forward pass through the model. The input 'x' is passed through the ResNet18 backbone
        followed by the custom classifier.

        Args:
            x (torch.Tensor): Input tensor (image) of shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: Output tensor with the class probabilities (shape: batch_size x output_classes).
        """
        return self.model(x)

# ===========================
# Section 2: Utility Functions
# ===========================

def process_image(image_path, device='cpu'):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a tensor.
    This function takes the path of an image, applies a series of preprocessing steps (scaling, cropping, 
    converting to tensor, and normalization), and returns a tensor ready for model input. The image will be 
    loaded from the specified file path, resized, and normalized to match the input expectations of models 
    trained on ImageNet (e.g., ResNet, VGG). The resulting tensor will be moved to the specified device 
    (CPU or GPU).
        Args:
        image_path (str): The file path to the image that needs to be processed.
        device (str or torch.device): The device to move the tensor to after processing. Default is 'cpu'. 
                                      If 'cuda' is specified, the tensor will be moved to GPU.

    Returns:
        torch.Tensor: A 4D tensor representing the processed image, with shape 
                      (1, 3, 224, 224) for a single image, ready to be fed into a model.
    """
    img = Image.open(image_path).convert("RGB")# Open and convert the image to RGB mode
    inference_transforms = transforms.Compose([
        transforms.Resize(256),# Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),# Crop the central 224x224 region
        transforms.ToTensor(), # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],# Normalize with ImageNet mean and std
                             std=[0.229, 0.224, 0.225])
    ])
     # Apply transformations The image is transformed by the inference_transforms pipeline, which resizes, crops, converts to a tensor, and normalizes the image.
    img_tensor = inference_transforms(img)
    # Add a batch dimension (1, 3, 224, 224) and move the tensor to the specified device
    img_tensor = img_tensor.unsqueeze(dim=0).to(device) #Adds an extra dimension to the tensor to make it suitable for batch processing, as PyTorch models expect a batch of images (even if it's just one).
    return img_tensor

def predict(model, image_path, top_k=3, device='cpu', cat_to_name=None):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
        This function processes an image, feeds it into the model, and returns the top K predicted classes 
    with their corresponding probabilities. The class labels can optionally be mapped to human-readable 
    names (if a `cat_to_name` dictionary is provided).

    Args:
        model (torch.nn.Module): The pre-trained deep learning model to use for prediction (e.g., ResNet, VGG).
        image_path (str): The path to the image for which predictions are to be made.
        top_k (int, optional): The number of top predictions to return. Default is 3.
        device (str or torch.device, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). 
                                                Default is 'cpu'.
        cat_to_name (dict, optional): A dictionary mapping class indices to human-readable class names. 
                                      If not provided, class indices will be used.
     Returns:
        str: A formatted string listing the top K predicted class names (or indices) and their probabilities.
    """
    model.to(device)
    model.eval()

    # Process input image (scale, crop, normalize)
    tensor_image = process_image(image_path, device=device)

    with torch.no_grad():
         # Get model's raw output for the image
        outputs = model(tensor_image)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)
  # Get the top K probabilities and corresponding indices
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
# Convert the top probabilities and indices to NumPy arrays (removing the batch dimension)
    # Remove batch dimension and convert to numpy
    top_probs = top_probs.detach().cpu().numpy()[0]
    top_indices = top_indices.detach().cpu().numpy()[0]

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    # Map to class names if provided
    # If 'cat_to_name' is provided, map the class labels to human-readable names
    if cat_to_name:
        top_class_names = [cat_to_name.get(cls, cls) for cls in top_classes]
    else:
        top_class_names = top_classes

    # Create a formatted string of predictions
    prediction_str = "Here are the top predictions based on the provided image:\n\n"
    for rank, (cls, prob) in enumerate(zip(top_class_names, top_probs), start=1):
        prediction_str += f"{rank}. {cls}: {prob * 100:.2f}%\n"
    return prediction_str

def load_checkpoint(filepath, device='cpu'):
    """
    Load a model checkpoint and rebuild the model.
    Loads a model checkpoint and reconstructs the model.

    This function loads a saved model checkpoint, including the model architecture, 
    model weights, class-to-index mapping, and other relevant information, and returns 
    the model along with the class-to-index dictionary.

    Args:
        filepath (str): The file path to the saved checkpoint.
        device (str or torch.device, optional): The device to load the model onto ('cpu' or 'cuda'). 
                                                Default is 'cpu'.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model loaded with the saved state dict.
            - class_to_idx (dict): A dictionary mapping class indices to class labels.
   
    """
    # Load checkpoint data from the provided file path
    checkpoint = torch.load(filepath, map_location=device)
    # Extract model architecture and other relevant data
    arch = checkpoint['arch']
    output_classes = checkpoint['output_classes']

     # Initialize the model based on the architecture type
    model = SkinDiseaseClassifier(
        arch=arch,
        output_classes=output_classes
    )
    # Load the saved model state dict (weights)
    model.load_state_dict(checkpoint['state_dict'])
    # Set class-to-index mapping for later use
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    class_to_idx = checkpoint['class_to_idx']
    return model,class_to_idx

def save_checkpoint(model, save_dir, class_to_idx, learning_rate, epochs):
    """
    Save the model checkpoint.
      This function saves the model architecture, weights, and other training parameters
    (e.g., learning rate, number of epochs) into a checkpoint file, and stores it in
    the given directory.
     Args:
        model (torch.nn.Module): The model to be saved.
        save_dir (str): The directory where the checkpoint will be saved.
        class_to_idx (dict): A dictionary mapping class indices to class labels.
        learning_rate (float): The learning rate used during training.
        epochs (int): The number of epochs the model was trained for.
    Returns:
        None: The checkpoint is saved as a file to the specified directory.
    
    """
    # Create checkpoint dictionary to store the model and training parameters
    checkpoint = {
        'arch': model.arch,
        'output_classes': model.output_classes,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs
    }
    # Ensure the directory exists, create if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
     # Define checkpoint file path
    save_path = os.path.join(save_dir, 'checkpoint.pth')
      # Save the checkpoint to the specified path
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

# ===========================
# Section 3: Data Loading Function
# ===========================

def load_data(data_dir, batch_size=16, valid_size=0.15, test_size=0.15):
    """
    Load data from data_dir, split into train, validation, and test datasets.
    Loads the image data from the specified directory and splits it into training, validation, 
    and test datasets, with appropriate transformations applied to each.
    This function applies different image transformations for training (e.g., data augmentation) 
    and for validation/test (e.g., resizing and normalization). The dataset is then split into 
    training, validation, and test subsets based on the given sizes.
    Args:
        data_dir (str): The directory containing the image dataset, with subdirectories for each class.
        batch_size (int, optional): The batch size for loading data. Default is 16.
        valid_size (float, optional): The fraction of data to use for validation. Default is 0.15.
        test_size (float, optional): The fraction of data to use for testing. Default is 0.15.
  Returns:
        tuple: A tuple containing:
            - trainloader (DataLoader): DataLoader for the training dataset.
            - validloader (DataLoader): DataLoader for the validation dataset.
            - testloader (DataLoader): DataLoader for the test dataset.
   
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224), # Random crop and resize to 224x224
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.RandomVerticalFlip(), # Random vertical flip
        transforms.RandomRotation(45),# Random rotation between -45 and 45 degrees
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),# Random color adjustments (brightness, contrast, etc.)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# Random affine transformation
        transforms.ToTensor(),# Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Normalize based on ImageNet statistics
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load the full dataset from the data directory using ImageFolder
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    # Calculate the sizes of the training, validation, and test sets
    total_size = len(full_dataset)
    test_len = int(test_size * total_size)
    valid_len = int(valid_size * total_size)
    train_len = total_size - test_len - valid_len
    # Split the dataset into training, validation, and test sets
    train_data, valid_data, test_data = random_split(full_dataset, [train_len, valid_len, test_len])

    # Apply transforms to validation and test transformations to the corresponding datasets
    valid_data.dataset.transform = valid_test_transforms
    test_data.dataset.transform = valid_test_transforms

    # Define dataloaders with drop_last=True for the training set Create DataLoaders for each dataset with the specified batch size
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return trainloader, validloader, testloader, full_dataset.class_to_idx

# ===========================
# Section 4: Calculate Class Weights
# ===========================

def calculate_class_weights(trainloader, device):
    """
    Calculate class weights to handle class imbalance.\
    This function iterates through the training data to extract the labels and then computes 
    class weights based on the frequency of each class. The weights are calculated using 
    the `compute_class_weight` method from scikit-learn and are returned as a tensor that 
    can be used in loss functions like `CrossEntropyLoss` to handle class imbalance during 
    model training.
     Args:
        trainloader (DataLoader): The DataLoader for the training dataset. It is used to 
                                  extract the labels of the images.
        device (str or torch.device): The device to move the class weights to ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor containing the class weights, moved to the specified device.
                      The shape is (num_classes,) where `num_classes` is the number of unique classes.
   
    """
    # Extract all labels from the training data
    labels = []
    for _, label in trainloader:
        labels.extend(label.numpy())
# Get unique classes
    classes = np.unique(labels)
    # Compute class weights using the balanced strategy from sklearn
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
     # Convert the class weights to a PyTorch tensor and move to the specified device
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# ===========================
# Section 5: Training Function
# ===========================

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device, save_dir):
    """
    Train the model and and evaluates it on the validation set after each epoch.
     This function trains the model on the training dataset and evaluates its performance on the validation dataset.
    It also tracks training loss, validation loss, accuracy, precision, recall, and F1 score. The learning rate 
    is adjusted during training using the ReduceLROnPlateau scheduler. Early stopping is implemented based on 
    validation loss to prevent overfitting.
    Args:
        model (torch.nn.Module): The model to train.
        trainloader (DataLoader): DataLoader for the training dataset.
        validloader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        epochs (int): The number of epochs to train the model.
        device (str or torch.device): The device to train the model on ('cpu' or 'cuda').
        save_dir (str): The directory to save the model checkpoint if the validation loss improves.
    Returns:
        None: The function trains the model and prints progress, but does not return anything.
   
    """
    # Initialize the learning rate scheduler and model on the given device
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    model.to(device)
    best_val_loss = float('inf')
    early_stopping_patience = 5  # Early stopping patience
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # Zero out the gradients from the previous step

            # Forward pass
            outputs = model(images)
            # Compute loss with L1 and L2 regularization
            ce_loss = criterion(outputs, labels)
            l1_lambda = 1e-5
            l2_lambda = 1e-4
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = ce_loss + l1_lambda * l1_norm + l2_lambda * l2_norm

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Update the progress bar with current loss and accuracy
            progress_bar.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{accuracy:.2f}%'})

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient calculation for validation
            for val_images, val_labels in validloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                ce_loss = criterion(val_outputs, val_labels)
                l1_lambda = 1e-5
                l2_lambda = 1e-4
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                v_loss = ce_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
                val_loss += v_loss.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

                all_preds.extend(val_predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
        # Track validation loss and accuracy
        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = val_loss / len(validloader)

        # Calculate additional metrics
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.3f}, Train Acc: {accuracy:.2f}% - "
              f"Val Loss: {avg_val_loss:.3f}, Val Acc: {val_accuracy:.2f}%\n"
              f"Val Precision: {val_precision:.3f}, Val Recall: {val_recall:.3f}, Val F1-score: {val_f1:.3f}\n")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model checkpoint
            checkpoint = {
                'arch': model.arch,
                'output_classes': model.output_classes,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epochs': epoch + 1
            }
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)['state_dict'])

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()

    # Plot validation F1-score
    plt.figure(figsize=(10, 5))
    plt.plot(val_f1s, label='Validation F1-score')
    plt.legend()
    plt.title('Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.show()

    return model

# ===========================
# Section 6: Testing Function
# ===========================

def test_model(model, testloader, criterion, device):
    """
    Evaluate the model on the test dataset and computes various performance metrics.
    This function evaluates the model's performance on the test dataset, including computing the 
    test loss, accuracy, precision, recall, and F1 score. The model is set to evaluation mode, 
    and no gradients are calculated during testing. Regularization terms (L1 and L2) are included 
    in the loss to ensure the model maintains regularization during inference.
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        testloader (DataLoader): The DataLoader for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (str or torch.device): The device to run the evaluation on ('cpu' or 'cuda').
    Returns:
        None: The function prints the evaluation results including loss, accuracy, precision, recall, 
              and F1 score, but does not return any value.
   
    """
    model.eval()
    # Initialize variables to accumulate test loss, correct predictions, and total samples
    test_loss = 0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_labels = []
  # Disable gradient calculation for inference
    with torch.no_grad():
          # Loop over batches of test data
        for test_images, test_labels in testloader:
            # Move data to the specified device
            test_images, test_labels = test_images.to(device), test_labels.to(device)
              # Forward pass
            test_outputs = model(test_images)
            # Compute the classification loss
            ce_loss = criterion(test_outputs, test_labels)
             # L1 and L2 regularization terms
            l1_lambda = 1e-5
            l2_lambda = 1e-4
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            t_loss = ce_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
             # Accumulate loss
            test_loss += t_loss.item()
                # Get predicted labels
            _, test_predicted = torch.max(test_outputs.data, 1)
             # Update total count and correct count
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
            # Store predictions and true labels for metric calculations
            all_preds.extend(test_predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())
     # Calculate accuracy
    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(testloader)

    # Calculate additional metrics
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Test Loss: {avg_test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Precision: {test_precision:.3f}, Test Recall: {test_recall:.3f}, Test F1-score: {test_f1:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ===========================
# Section 7: Main Execution Flow
# ===========================

def get_input_args():
    parser = argparse.ArgumentParser(description="Train or predict using a neural network on a dataset.")

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Sub-parser for the 'train' command
    train_parser = subparsers.add_parser('train', help='Train the neural network')
    train_parser.add_argument('data_dir', type=str, help='Directory containing the training data')
    train_parser.add_argument('--save_dir', type=str, default='./', help='Directory for saving checkpoints')
    train_parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture to use (default: resnet18)')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    train_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs for training')  # Changed default to 12
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    train_parser.add_argument('--category_names', type=str, default='./skin_disease_class_to_name.json', help='Path to category names JSON file')
    train_parser.add_argument('--top_k', type=int, default=3, help='Top K classes to display')

    return parser.parse_args()

def main():
    # Parse input arguments
    args = get_input_args()

    # Set the device to use GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.command == 'train':
        # Training process started
        print("Training process started...")

        # Load data
        print("Loading data...")
        trainloader, validloader, testloader, class_to_idx = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        print(f"Number of classes: {len(class_to_idx)}")

        # Initialize the model
        print("Initializing the model...")
        output_classes = len(class_to_idx)
        model = SkinDiseaseClassifier(
            arch=args.arch,
            output_classes=output_classes
        )
        model.class_to_idx = class_to_idx

        # Move model to the device
        model.to(device)

        # Calculate class weights
        print("Calculating class weights...")
        class_weights = calculate_class_weights(trainloader, device)
        print(f"Class weights: {class_weights}")

        # Define loss criterion with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Define optimizer (only parameters that require gradients)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate
        )

        # Train the model
        print("Starting training...")
        model = train_model(
            model=model,
            trainloader=trainloader,
            validloader=validloader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            save_dir=args.save_dir  # Pass save_dir here
        )

        # Test the model
        print("Testing the model...")
        test_model(
            model=model,
            testloader=testloader,
            criterion=criterion,
            device=device
        )

        print("Training completed successfully.")

    else:
        print("Please provide a valid command (e.g., 'train').")

if __name__ == '__main__':
    main()

# ===========================
# Section 8: Make Predictions
# ===========================

def make_prediction(image_path, device, save_dir, category_names, top_k):
    """
    Make a prediction for a single image. for a single image using a trained model.

    This function loads the trained model from the saved checkpoint, processes the input image, 
    and makes a prediction. It returns the top K predicted class labels along with their corresponding 
    probabilities. Optionally, it can map the class indices to human-readable category names if a 
    `category_names` file is provided.
    Args:
        image_path (str): The file path to the image for which a prediction is to be made.
        device (str or torch.device): The device to run the model on ('cpu' or 'cuda').
        save_dir (str): The directory where the saved model checkpoint is located.
        category_names (str): The file path to a JSON file that maps class indices to category names. 
                              If not provided, class indices will be used.
        top_k (int): The number of top predictions to return.
     Returns:
        None: The function prints the top K predictions (class names or indices and their probabilities) 
              to the console, but does not return any value.
   
    """
    # Path to the saved checkpoint
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')

    # Load the model from checkpoint
    model = load_checkpoint(checkpoint_path, device=device)

    # Load category names if provided
    if os.path.exists(category_names):
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = {}

    # Make prediction
    prediction = predict(model, image_path, top_k=top_k, device=device, cat_to_name=cat_to_name)

    print(prediction)


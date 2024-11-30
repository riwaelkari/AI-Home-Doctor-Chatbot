# Chatbot Home Doctor

A chatbot application that provides medical assistance and advice.

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Symptom Checker**: Input symptoms to receive potential disease diagnoses.
- **Skin Disease Diagnosis**: Upload images to analyze and identify possible skin conditions.
- **Conversational Interface**: Interactive chatbot powered by LangChain for seamless user experience.
- **Secretary Assistant**: "Donna the Secretary" handles scheduling and administrative tasks.
- **Image and Audio Support**: Accepts and processes images and audio files for comprehensive assistance.

## Datasets

This project utilizes the following datasets:

1. **Skin Disease Dataset**
   - **Description**: A collection of images categorized by various skin diseases.
   - **Source**: [Kaggle - Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
   - **Usage**: Used to train the skin disease prediction model.

2. **Symptom Disease Dataset**
   - **Description**: Contains symptom descriptions and associated diseases.
   - **Source**: [Kaggle - Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
   - **Usage**: Utilized for the symptom-based disease prediction model.

**Note**: Please ensure you have downloaded these datasets from Kaggle and place them in the appropriate directories as specified in the [Project Structure](#project-structure) section.

## Project Structure

- `actual_models\`: Contains the files used to train the models, including the prediction functions.
- `chatbot\`: Includes the agent that runs the chatbot logic and all the langchain chains.
- `chatbot\chains\`: Contains all the chains used for the base chain, symptom disease doctor, skin disease doctor, and donna the secretary.
- `chatbot_index`: Contains the files needed to run the Similarity Search.
- `dataset\`: Contains the datasets for Symptom disease doctor only.
- `frontend\`: Contains the frontend codes of the Flask server including html, css, and js.
- `saved_models\`: Contains the checkpoint file for skin disease and knn model.
- `server\`: Contains the Flask server setup and setup for all the chains.
- `uploads\`: Folder to contain recieved images and audios.
- `Dockerfile`: For containerizing the application.
- `requirements.txt`: Python dependencies.

## Setup Instructions

Setting up the **Chatbot Home Doctor** involves several steps, including environment setup, dependency installation, dataset preparation, and running the application. Follow the steps below to get your project up and running.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Docker** (optional, for containerization): [Download Docker](https://www.docker.com/get-started)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Kaggle Account** (optional, to visualize the datsets used): To download the datasets.

### Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/yourusername/chatbot-home-doctor.git
   cd chatbot-home-doctor
   ```

2. **Setup a Virtual Environment (Recommended)**

   ```
   python -m venv venv
   venv\Scripts\activate # For windows
   ```

3. **Install Dependencies**

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Configuration

1. **Add Environment Variables**

Creat a `.env` file in the root directory and add the following neseccary environmental variables:

   ```
   SECRET_TOKEN = '' # Your OpenAI API Key 
   GMAIL_PASS = '' # Password of the account used to send reminders
   GMAIL_USER = '' # email of the account used to send reminders
   ```

### Run the Application

1. **Start the Flask server**

Open a terminal in the root repository and run the following:

   ```
   python -m server.server
   ```
The server should start on `http://localhost:5000/`.



# Chatbot Home Doctor

A chatbot application that provides medical assistance and advice.

<img src="frontend/images/symptom_disease_icon.png" alt="Symptom Disease Icon" width="200" height="200" />

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Configuration and Running](#configuration-and-running)
- [Meet the Team](#meet-the-team)
- [Acknowledgements](#acknowledgements)

## Features

- **Hierarchical Structure**: Utilizes Hierarchy in which the agent controls which chain will be selected based on user input.
- **Symptom Disease Doctor**: Input symptoms to receive potential disease diagnoses and insights.
- **Skin Disease Doctor**: Upload images to analyze and identify possible skin conditions.
- **Secretary Assistant**: Donna the Secretary handles medication scheduling for reminders.
- **Seamless Integration**: Supports very simple integration of additional models, especially due to the versatile frontend and protected Chatbot API calls passing through Guards.
- **Conversational Interface**: Interactive chatbot powered by LangChain for seamless user experience.
- **Image and Audio Support**: Accepts and processes images and audio files for comprehensive assistance.
- **Bilingual Support**: Accepts both English and Arabic.
- **Query refinement**: Uses query refiners to make sure the chatbot gets the right user input.
- **Guardrail Support**: Utilizes Guards to protect the user from unintended chatbot answers.

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

## Project Structure

- `actual_models\`: Contains the files used to train the models, including the prediction functions.
- `chatbot\`: Includes the agent that runs the chatbot logic and all the langchain chains.
- `chatbot\chains\`: Contains all the chains used for the base chain, symptom disease doctor, skin disease doctor, and donna the secretary.
- `chatbot_index`: Contains the files needed to run the Similarity Search.
- `dataset\`: Contains the datasets for Symptom disease doctor only.
- `frontend\`: Contains the frontend codes of the Flask server including html, css, and js.
- `saved_models\`: Contains the checkpoint file for skin disease and knn model.
- `server\`: Contains the Flask server setup and setup for all the chains.
- `uploads\`: Folder to contain received images and audios.
- `Dockerfile`: For containerizing the application.
- `requirements.txt`: Python dependencies.

## Setup Instructions

Setting up the **Chatbot Home Doctor** involves several steps, including environment setup, dependency installation, dataset preparation, and running the application. Follow the steps below to get your project up and running.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Docker** (Recommended, for containerization): [Download Docker](https://www.docker.com/get-started)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Kaggle Account** (optional, to visualize the daatsets used): To download the datasets.

### Configuration and Running
1. **Clone the Repository**

   ```
   git clone https://github.com/dayehhadi/AI-Home-Doctor-Chatbot
   cd AI-Home-Doctor-Chatbot
   ```
2. **Setup a Virtual Environment (Recommended)**

   ```
   python -m venv venv
   venv\Scripts\activate # For windows
   ```

3. **Add Environment Variables**

In Your DockerFile fill the folllowing:

   ```
   ENV SECRET_TOKEN = '' # Your OpenAI API Key 
   ENV GMAIL_PASS = '' # Password of the account used to send reminders
   ENV GMAIL_USER = '' # email of the account used to send reminders
   ```

4. **Run a Docker Image**

In the directory that has the Dockerfile, run the following:
   ```
   docker build -t chatbot-home-doctor .

   docker run -p 5000:5000 --name chatbot-container chatbot-home-doctor

   #or run from docker and use port 5000 
   ```
5. **Starting the Flask server**

THe Dockerfile opens a terminal in the root repository and run the following:

   ```
   python -m server.server
   ```
The server should start on `http://127.0.0.1:5000/`.

## Meet the Team 

Hadi Dayeh

Nour Shammaa

Zeinab Mazraani

Riwa El Kari

## Acknowledgements

- Special Thanks to Professor Ammar Mhanna for providing us with the OpenAI API Key, without him we could not have implemented such a good chatbot.
- [LangChain](https://www.langchain.com/) for the conversational AI framework.
- [Kaggle](https://www.kaggle.com/) for providing the datasets.
- [OpenAI](https://openai.com/) for the GPT-4 architecture powering the chatbot.

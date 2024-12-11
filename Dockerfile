# Use an official Python runtime as the base image
FROM python:3.9-slim


# Set environment variables to prevent Python from writing .pyc files and to buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SECRET_TOKEN=''
ENV GMAIL_PASS='sgcm dbnm dohx ctdg'
ENV GMAIL_USER='donasecretary@gmail.com'

# Set the working directory inside the container
WORKDIR /app


# Install system dependencies
# (Optional: Uncomment the following lines if your project requires additional system packages)



RUN apt-get update && apt-get install -y \
     build-essential \
     gcc \
     python3.dev \
     libpcap-dev \
     libssl-dev \
     libffi-dev \
     libpq-dev \
     ffmpeg \
     && rm -rf /var/lib/apt/lists/*


# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install gitpython
RUN pip install -r requirements.txt


# Copy project
COPY . /app/


# Expose port
EXPOSE 5000


# Run the application
CMD ["python", "-m", "server.server"]
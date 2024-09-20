# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Expose port (if your app uses a specific port)
EXPOSE 5000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app/main.py"]

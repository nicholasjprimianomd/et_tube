# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the ML model file
#COPY production.pth /app/model/

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Assuming you have a line to install other dependencies, add libglib2.0-0 to it
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    certbot \
    gunicorn \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make port 80 available to the world outside this container
EXPOSE 80

EXPOSE 443

EXPOSE 5000

# Define environment variable
ENV NAME World

# Environment variable to define the model path
ENV MODEL_PATH /app/model/production.pth

# Run app.py when the container launches
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]


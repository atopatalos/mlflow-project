# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container (change this to docker app)
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the MLflow server port
EXPOSE 80

# Run a custom script to start the fastAPI server
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]

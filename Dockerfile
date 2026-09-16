# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the live server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
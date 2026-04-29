# Start from an official Python image
# This gives us a clean Linux computer with Python already installed
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (Docker caches this layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Tell Docker that the app will use port 8000
EXPOSE 8000

# Command to start the API when container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
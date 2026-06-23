# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg is required for pydub and whisper audio processing
# git and build-essential are needed for compiling package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# We install torch and torchaudio with CPU support to keep the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories for storage and database
RUN mkdir -p data/raw data/processed data/output data/database uploads

# Expose port for FastAPI REST API server
EXPOSE 8000

# Default command starts the API server
CMD ["python", "run.py", "api", "--host", "0.0.0.0", "--port", "8000"]

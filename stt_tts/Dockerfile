#docker buildx build --platform=linux/arm64 -t stt_tts .
#docker run -p 8000:8000 stt_tts

# Use Python 3.8 base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Update package lists and install necessary packages
RUN apt-get update -o Acquire::AllowInsecureRepositories=true \
    && apt-get install -y --no-install-recommends \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone the speechbrain repository
RUN git clone --branch v1.0.2 https://github.com/speechbrain/speechbrain.git /speechbrain --depth 1
WORKDIR /speechbrain
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Go back to the app directory
WORKDIR /app

# Copy the application code
COPY app.py app.py

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
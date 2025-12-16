#docker buildx build --platform=linux/arm64 -t embedding .
#docker run -p 8080:8080 embedding

FROM python:3.9
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Go back to the app directory
WORKDIR /app

# Copy the application code
COPY app.py app.py

ENV PYTHONWARNINGS="ignore::FutureWarning"

# Expose port 8080
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
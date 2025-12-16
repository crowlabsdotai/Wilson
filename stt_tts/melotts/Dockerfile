#docker buildx build --platform=linux/arm64 -t melotts .
#docker run -p 8888:8888 melotts

FROM python:3.9
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/myshell-ai/MeloTTS.git /MeloTTS
WORKDIR /MeloTTS
RUN pip install -e .
RUN python -m unidic download

RUN pip install --upgrade botocore boto3 s3transfer

# Go back to the app directory
WORKDIR /app

# Copy the application code
COPY app.py app.py

ENV PYTHONWARNINGS="ignore::FutureWarning"

RUN python -c "import nltk; nltk.download('all')"
RUN pip uninstall torchvision

# Expose port 8888
EXPOSE 8888

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888"]
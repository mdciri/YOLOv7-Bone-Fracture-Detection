# app/Dockerfile

FROM python:3.9

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    cmake \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir app
WORKDIR /app

COPY webapp.py .
COPY ../yolov7-p6-bonefracture.onnx .

COPY requirements-docker.txt .
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements-docker.txt

ENTRYPOINT ["streamlit", "run", "webapp.py", "--server.port=8501", "--server.address=0.0.0.0"]
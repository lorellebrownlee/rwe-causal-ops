FROM python:3.11-slim

WORKDIR /app

# Install git (needed by DVC) and dependencies
RUN apt-get update && apt-get install -y git libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN git config --global user.email "ci@docker" && \
    git config --global user.name "Docker" && \
    git init && \
    git add -A && \
    git commit -m "docker build snapshot" --allow-empty

RUN mkdir -p data/raw data/interim results reports

EXPOSE 5000

CMD ["bash", "-c", "dvc repro && mlflow ui --host 0.0.0.0 --port 5000"]
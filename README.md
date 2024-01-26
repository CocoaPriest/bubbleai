# BubbleAI

BubbleAI is a sophisticated AI-driven platform designed to ingest, process, and analyze documents, leveraging the power of OpenAI's embeddings and a robust FastAPI backend. It's built to handle a variety of tasks, including document ingestion, text extraction, and querying for insights using advanced natural language processing techniques.
BubbleAI serves as the backend for [AssistAI](https://github.com/CocoaPriest/AssistAI), a macOS application that utilizes its endpoints for RAG, ingestion and interference.

## Features

-   **Document Ingestion**: Securely upload documents to be processed and analyzed.
-   **Text Extraction**: Extract and process text from a variety of document formats.
-   **Vector Embedding**: Utilize OpenAI's embeddings to convert text into vector representations for advanced analysis.
-   **Query System**: Ask questions and receive insights based on the ingested and processed documents.
-   **Scalable Architecture**: Deployed using Docker and AWS services for scalability and reliability.

## Stack

-   FastAPI
-   Uvicorn (server)
-   Gunicorn (process manager for Uvicorn)
-   NGINX forwards 80 to 8000 (uvicorn)
-   AWS CodeDeploy

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   AWS CLI configured with access to S3, SQS, and ECR
-   Python 3.11.3 or higher
-   An OpenAI API key

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/CocoaPriest/bubbleai.git
cd bubbleai
```

2. **Set up environment variables**

Create a `.env` file in the root directory and populate it with your AWS and OpenAI credentials:

```plaintext
OPENAI_API_KEY=your_openai_api_key
aws_access_key_id=your_aws_access_key_id
aws_secret_access_key=your_aws_secret_access_key
aws_region=your_aws_region
DATABASE_URL=your_database_url
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Build and run the Docker containers**

```bash
docker-compose up --build
```

### Usage

#### Ingesting Documents

To ingest a document, send a POST request to `/ingest` with the document file, its full path, and the machine ID.

```bash
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/document.pdf;type=application/pdf' \
  -F 'full_path=/documents/document.pdf' \
  -F 'machine_id=your_machine_id'
```

#### Asking Questions

To ask a question based on the ingested documents, send a POST request to `/ask` with your question.

```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the main topic of the ingested documents?"}'
```

### Deployment

The provided `Makefile` includes commands for AWS login, Docker login, and deployment. Use these commands to deploy the application to an AWS EC2 instance.

```bash
make login
make docker_login
make docker_deploy
```

### SSH

```bash
ssh -i ~/.ssh/fastapi_key.pem ubuntu@ec2-3-121-186-17.eu-central-1.compute.amazonaws.com
```

### NGINX

`sudo vim /etc/nginx/sites-enabled/fastapi_nginx`

## Docker on ec2

#### Pull

`docker pull 246532218018.dkr.ecr.eu-central-1.amazonaws.com/fastapi-bubble:latest`

#### Run

`docker run -t -p 80:8000 246532218018.dkr.ecr.eu-central-1.amazonaws.com/fastapi-bubble:latest`

### Run locally

1. `source ~/.virtualenvs/bubble/bin/activate.fish`
2. Start docker: `make docker_up`
   2a. Now, I want to stop docker's web and start my local:
   a. `docker stop web`
   b. `make local_web`

#### Login into

`docker exec -it [name/id] /bin/bash`

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

This README is a brief overview of BubbleAI. For more detailed documentation, please refer to the individual files and code comments within the project.

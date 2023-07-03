import json
import os
import uuid
from typing import Annotated, List

import asyncpg
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.asyncpg import register_vector
from pydantic import BaseModel

from logger import logger

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("aws_access_key_id")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("aws_secret_access_key")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("aws_region")

if os.environ.get("DATABASE_URL") is None:
    connection_string = os.getenv("DATABASE_URL")
else:
    connection_string = os.environ["DATABASE_URL"]
logger.info(f"postgres connection: {connection_string}")


s3bucket = "bubbleai.uploads"

app = FastAPI(title="BubbleAI")

embeddings = OpenAIEmbeddings()
s3 = boto3.resource("s3")
sqs = boto3.client("sqs")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResourceToDelete(BaseModel):
    uri: str
    machine_id: str


class Question(BaseModel):
    question: str


def send_sqs(message) -> bool:
    # FIFO Queue URL
    queue_url = "https://sqs.eu-central-1.amazonaws.com/246532218018/bubble-ingest.fifo"

    # TODO: get from JWT
    client_id = uuid.UUID(int=0)

    logger.info(f"Sending SQS message: `{message}`")

    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=message,
        MessageGroupId=str(client_id),
        MessageDeduplicationId=str(uuid.uuid4()),
    )

    return True


@app.get("/", summary="Root")
async def read_root():
    """
    Create an item with all the information:

    - **message**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """

    logger.info("Calling root")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    chunks = text_splitter.split_text(
        "Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well."
    )

    return {"message": chunks}


@app.get("/items/{item_id}", response_description="Something")
def get_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put(
    "/ingest",
    summary="Saves a file to s3",
    status_code=status.HTTP_201_CREATED,
)
def ingest(
    file: Annotated[UploadFile, File()],
    full_path: Annotated[str, Form()],
    machine_id: Annotated[str, Form()],
):
    logger.info(f"Ingesting... uri: `{full_path}`, machine_id: `{machine_id}`")

    # TODO: get from JWT
    client_id = uuid.UUID(int=0)

    logger.info(f"uploading file to s3: `{file.filename}`")
    key = f"{client_id}/{uuid.uuid4()}"

    ret = s3.Bucket(s3bucket).put_object(
        Key=key,
        Body=file.file,
        ContentType=file.content_type,
        Metadata={"full_path": full_path, "machine_id": machine_id},
    )

    logger.info(f"s3 object created: {ret.key}")
    sqs_message = {"action": "INGEST", "bucket": "bubbleai.uploads", "key": key}
    message = json.dumps(sqs_message)
    send_sqs(message)

    return {
        "file_name": file.filename,
        "content_type": file.content_type,
        "full_path": full_path,
        "machine_id": machine_id,
    }


@app.delete(
    "/resource",
    summary="Removes document from the index",
    status_code=status.HTTP_200_OK,
)
def remove_from_index(resource: ResourceToDelete):
    logger.info(f"Removing uri: `{resource.uri}`, machine_id: `{resource.machine_id}`")

    resource_dict = dict(resource)
    resource_dict["action"] = "DELETE"

    message = json.dumps(resource_dict)

    send_sqs(message)

    return resource_dict


@app.post("/ask", summary="Ask a question", status_code=status.HTTP_200_OK)
async def ask(question: Question):
    logger.info(f"Question: `{question.question}`")

    vector = embeddings.embed_query(question.question)
    # logger.info(f"Vector: {vector}")

    chunks = await cosineChunks(vector)
    logger.info(f"chunks: {chunks}")

    # TODO: create prompt, send to OpenAI (with functions??)

    return {"answer": "FINAL ANSWER", "chunks": chunks}


async def cosineChunks(vector: List[float]):
    async with asyncpg.create_pool(connection_string) as pool:
        async with pool.acquire() as conn:
            await register_vector(conn)

            query = """SELECT c.id, d.machine_id, d.full_path, c.text
                            FROM chunks AS c
                            INNER JOIN documents AS d
                                ON c.document_id=d.id
                            ORDER BY c.embedding <=> $1
                            LIMIT 5;"""

            result = await conn.fetch(query, vector)

            items = [
                {
                    "chunk_id": chunk_id,
                    "machine_id": machine_id,
                    "full_path": full_path,
                    "text": text,
                }
                for chunk_id, machine_id, full_path, text in result
            ]

            jsonItems = json.dumps(items)
            return items

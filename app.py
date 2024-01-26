import json
import os
import uuid
from typing import Annotated, List

import asyncpg
import boto3
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.asyncpg import register_vector
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from logger import logger

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
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
    is_folder: bool


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
    logger.info(
        f"Removing uri: `{resource.uri}`, machine_id: `{resource.machine_id}`, is_folder: `{resource.is_folder}`"
    )

    resource_dict = dict(resource)
    resource_dict["action"] = "DELETE"

    message = json.dumps(resource_dict)

    send_sqs(message)

    

    return resource_dict


@app.get(
    "/is_ingester_running",
    summary="Tells if remote ingester for this client is running",
    status_code=status.HTTP_200_OK,
)
def is_ingester_running():
    # TODO: get from JWT
    client_id = uuid.UUID(int=0)

    bucket = s3.Bucket(s3bucket)

    for obj in bucket.objects.filter(Prefix=str(client_id)):
        print(f"The folder {client_id} in the bucket {s3bucket} contains files.")
        return {"value": True}

    logger.info(f"The folder {client_id} in the bucket {s3bucket} is empty.")
    return {"value": False}


@app.post("/ask", summary="Ask a question", status_code=status.HTTP_200_OK)
async def ask(question: Question, request: Request):
    logger.info(f"Question: `{question.question}`")

    vector = embeddings.embed_query(question.question)
    # logger.info(f"Vector: {vector}")

    chunks = await cosine_chunks(vector)
    logger.info(f"chunks: {chunks}")

    user_prompt = get_user_prompt(question=question.question, chunks=chunks)
    # logger.info(f"User prompt:\n{user_prompt}")

    system_prompt = get_system_prompt()
    # logger.info(f"System prompt:\n{system_prompt}")

    event_generator = get_answer_event_generator(request, system_prompt, user_prompt)
    return EventSourceResponse(event_generator)


async def cosine_chunks(vector: List[float]):
    async with asyncpg.create_pool(connection_string) as pool:
        async with pool.acquire() as conn:
            await register_vector(conn)

            async with conn.transaction():
                # await conn.execute("SET LOCAL ivfflat.probes = 10;")

                query = """SELECT c.id, d.machine_id, d.full_path, c.text
                                FROM chunks AS c
                                INNER JOIN documents AS d
                                    ON c.document_id=d.id
                                ORDER BY c.embedding <-> $1
                                LIMIT 10;"""

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

                return items


def get_system_prompt() -> str:
    with open("prompts/system.txt", "r") as f:
        template = f.read()
    return template


def get_user_prompt(question: str, chunks: List[dict[str:str]]) -> str:
    with open("prompts/user.txt", "r") as f:
        template = f.read()

    summaries = ""
    for chunk in chunks:
        summaries += "<CHUNK>" + chunk["text"] + "</CHUNK>\n"
        summaries += (
            "<SOURCE>" + chunk["machine_id"] + "@" + chunk["full_path"] + "</SOURCE>\n"
        )
        summaries += "\n"
    prompt = template.format(question=question, summaries=summaries)
    return prompt


async def get_answer_event_generator(
    request: Request, system_prompt: str, user_prompt: str
):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    collected_messages = []
    hidden_buffer1_triggered = False
    hidden_buffer2_triggered = False

    for chunk in response:
        if await request.is_disconnected():
            logger.warning("Request disconnected")
            # TODO: send cancel request to OpenAI
            return

        # logger.info(f"Chunk: {chunk}")
        chunk_message = chunk["choices"][0]["delta"]  # extract the message
        collected_messages.append(chunk_message)  # save the message

        content = chunk_message.get("content", "")
        # logger.info(f"Content: {content}")

        if content == ">>":
            hidden_buffer1_triggered = True
        elif hidden_buffer1_triggered == True and content == "||":
            hidden_buffer2_triggered = True
        elif hidden_buffer1_triggered and hidden_buffer2_triggered:
            yield {
                "event": "src_upd",
                "retry": 30000,
                "data": content,
            }
        else:
            yield {
                "event": "answ_upd",
                "retry": 30000,
                "data": content,
            }

    # print the time delay and text received
    full_reply_content = "".join([m.get("content", "") for m in collected_messages])
    logger.info(f"Full conversation received: {full_reply_content}")

    yield {"event": "end", "data": ""}

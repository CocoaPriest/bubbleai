from pydantic import BaseModel
from fastapi import FastAPI, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import Annotated
from logger import logger
import boto3
import uuid
import os

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("aws_access_key_id")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("aws_secret_access_key")
os.environ["AWS_REGION"] = os.getenv("aws_region")

app = FastAPI(title="BubbleAI")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class File2(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


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
    s3 = boto3.resource("s3")

    # TODO: get from JWT
    client_id = uuid.UUID(int=0)

    logger.info(f"uploading file to s3: `{file.filename}`")

    ret = s3.Bucket("bubbleai.uploads").put_object(
        Key=f"{client_id}/{uuid.uuid4()}",
        Body=file.file,
        ContentType=file.content_type,
        Metadata={"full_path": full_path, "machine_id": machine_id},
    )
    logger.info(f"s3 object created: {ret.key}")
    return {
        "file_name": file.filename,
        "content_type": file.content_type,
        "full_path": full_path,
        "machine_id": machine_id,
    }

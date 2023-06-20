import boto3
import json
from langchain.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.docstore.document import Document
from typing import List, Tuple

from psycopg.conninfo import make_conninfo

import os
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("aws_access_key_id")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("aws_secret_access_key")
os.environ["AWS_REGION"] = os.getenv("aws_region")

embeddings = OpenAIEmbeddings()

# connection_string = PGVector.connection_string_from_db_params(
#     driver=os.environ.get("DB_DRIVER", "psycopg"),
#     host=os.getenv("DB_HOST"),
#     port=os.getenv("DB_PORT"),
#     database=os.getenv("DB_DATABASE"),
#     user=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASSWORD"),
# )

connection_string = make_conninfo(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

print(f"=> postgres connection: {connection_string}")


def receive_messages_from_sqs_in_batches(queue_url):
    # Create SQS client
    sqs = boto3.client("sqs")

    # Implement the long polling mechanism
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=["All"],
            MaxNumberOfMessages=10,
            WaitTimeSeconds=7,  # Longer polling up to 20 seconds
            VisibilityTimeout=15,  # Increase this as needed
        )

        # make sure to increase the `VisibilityTimeout` parameter if my processing
        # function might take more than 30 seconds to avoid the same message being
        # sent to another consumer before it's deleted.

        # Check if any messages are received
        if "Messages" in response:
            for message in response["Messages"]:
                # Process the message
                print(f"message type: {type(message)}")
                documents = load_documents(message)
                chunks = split_documents(documents)

                texts = [chunk.page_content for chunk in chunks]
                vectors = embeddings.embed_documents(texts)
                zipped = zip(texts, vectors)

                try:
                    persist(zipped, "machine_id", "full_path")  # TODO
                except:
                    print(f"=> not gonna delete SQS message")
                else:
                    # Delete the message from the queue
                    sqs.delete_message(
                        QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
                    )
                    print(f"=> SQS message deleted")
                    # TODO: delete file from s3


def load_documents(message: any) -> List[Document]:
    content = json.loads(message["Body"])

    # Get S3 object details from the S3 event
    bucket = content["Records"][0]["s3"]["bucket"]["name"]
    key = content["Records"][0]["s3"]["object"]["key"]

    print(f"=> Processing bucket {bucket}, key: {key}")

    # TODO: no, looks like I have to load files manually with boto3,
    # just to be flexible with docuemnt loaders. First, check it in a notebook!
    # also, to read file's metadata like file_path
    loader = S3FileLoader(bucket, key)
    documents = loader.load()
    print(f"=> Document:\n {documents}")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    print(f"=> Chunks:\n {len(chunks)}")
    return chunks


def persist(zipped: List[Tuple[str, List[float]]], machine_id: str, full_path: str):
    print("=> Saving to postgres...")

    try:
        with psycopg.connect(connection_string) as conn:
            register_vector(conn)

            # TODO: batch insert
            for text, vector in zipped:
                insert_query = "INSERT INTO documents (embedding, text, machine_id, full_path) VALUES (%s, %s, %s, %s)"
                conn.execute(insert_query, (vector, text, machine_id, full_path))
            conn.commit()
    except psycopg.Error as e:
        print(f"An error occurred: {e}")
        raise


# Replace with your queue URL
queue_url = "https://sqs.eu-central-1.amazonaws.com/246532218018/bubble-ingest"

print(f"=> running ingester for SQS: `{queue_url}`")
receive_messages_from_sqs_in_batches(queue_url)

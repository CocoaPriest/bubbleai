import boto3
import json
from logger import logger
from langchain.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.docstore.document import Document
from typing import List, Tuple, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
import tempfile

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

s3client = boto3.client("s3")
sqs = boto3.client("sqs")

embeddings = OpenAIEmbeddings()

connection_string = make_conninfo(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

logger.info(f"postgres connection: {connection_string}")


def receive_messages_from_sqs_in_batches(queue_url):
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
                logger.info(f"message type: {type(message)}")
                document = load_document(message)
                chunks = split_document(document)
                texts = [chunk.page_content for chunk in chunks]
                logger.info(f"chunks:\n{chunks}")

                vectors = embeddings.embed_documents(texts)
                zipped = zip(texts, vectors)

                try:
                    logger.info(f"Document metadata:\n{document.metadata}")
                    raise
                    persist(zipped, document.metadata)  # TODO
                except:
                    logger.info(f"not gonna delete SQS message")
                else:
                    # Delete the message from the queue
                    sqs.delete_message(
                        QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
                    )
                    logger.info(f"SQS message deleted")
                    # TODO: delete file from s3


def load_document(message: any) -> Document:
    content = json.loads(message["Body"])

    # Get S3 object details from the S3 event
    bucket = content["Records"][0]["s3"]["bucket"]["name"]
    key = content["Records"][0]["s3"]["object"]["key"]

    logger.info(f"Processing bucket {bucket}, key: {key}")

    # I have to load files manually with boto3,
    # just to be flexible with docuemnt loaders.
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/{key}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            s3client.download_file(bucket, key, file_path)
            logger.info(f"File {key} has been downloaded successfully to {file_path}")

            s3metadata = get_s3metadata(bucket, key)
            elements = get_elements(file_path, s3metadata["content_type"])
            text = "\n\n".join([str(el) for el in elements])

            doc = Document(page_content=text, metadata=s3metadata)

            return doc

        except Exception as e:
            logger.error(
                f"There was an error while downloading the file {key} from the bucket {bucket}."
            )
            logger.error(f"{e}")


def get_elements(file_path, content_type) -> List[Element]:
    # not using auto, arguments: https://unstructured-io.github.io/unstructured/bricks.html
    if content_type == "application/pdf":
        # other `strategy` options produce worse results
        return partition_pdf(file_path, strategy="fast")
    else:
        logger.error(f"Unknown content_type: {content_type}")
        raise


def get_s3metadata(bucket, key):
    # Head the object
    response = s3client.head_object(Bucket=bucket, Key=key)

    # The 'Metadata' field is a dictionary of the user metadata
    metadata = response["Metadata"]
    metadata["content_type"] = response["ContentType"]
    logger.info(f"metadata:\n{metadata}")
    return metadata


def split_document(document: Document) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs: List[Document] = list()
    docs = [document]
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Number of chunks: {len(chunks)}")
    return chunks


# GPT4:
# Your code is adding inserts into the database within the loop one by one. Since you're seeking
# to perform a batch insert instead of a looped, one-by-one insert, you could use the `executemany`
# command from the psycopg2 library. Here's how you might modify your code to achieve this:

# ```python
# with psycopg.connect(connection_string) as conn:
#     register_vector(conn)

#     # Prepare data for batch insert
#     data = [(vector, text, machine_id, full_path) for text, vector in zipped]

#     insert_query = """
#         INSERT INTO documents (embedding, text, machine_id, full_path)
#         VALUES (%s, %s, %s, %s)
#     """

#     # Execute the batch insert
#     with conn.cursor() as cur:
#         cur.executemany(insert_query, data)

#     conn.commit()
# ```

# Here, `data` is a list of tuples where each tuple corresponds to a row to be inserted.
# The `executemany` function is able to take this list and perform a batch insert, which can be
# more efficient than separate insert commands.

# Please note that this form of batch insertion is preferred when you're sure that your list
# of data isn't huge.


def persist(
    zipped: List[Tuple[str, List[float]]],
    metadata: Dict[str, str],
):
    logger.info("Saving to postgres...")
    machine_id = metadata["machine_id"]
    full_path = metadata["full_path"]
    content_type = metadata["content_type"]

    try:
        with psycopg.connect(connection_string) as conn:
            register_vector(conn)

            # TODO: normalize: use a separate table for documents and chunks
            # TODO: batch insert with asyncpg
            for text, vector in zipped:
                insert_query = "INSERT INTO documents (embedding, text, machine_id, full_path) VALUES (%s, %s, %s, %s)"
                conn.execute(insert_query, (vector, text, machine_id, full_path))
            conn.commit()
    except psycopg.Error as e:
        logger.error(f"An error occurred: {e}")
        raise


# Replace with your queue URL
queue_url = "https://sqs.eu-central-1.amazonaws.com/246532218018/bubble-ingest"

logger.info(f"running ingester for SQS: `{queue_url}`")
receive_messages_from_sqs_in_batches(queue_url)

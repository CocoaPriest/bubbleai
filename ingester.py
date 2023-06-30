import boto3
from botocore.exceptions import ClientError
import json
import tempfile
import os
from ContentTypeException import ContentTypeException
from logger import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
import asyncio
import asyncpg
from asyncpg import exceptions
from pgvector.asyncpg import register_vector

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("aws_access_key_id")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("aws_secret_access_key")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("aws_region")

s3 = boto3.client("s3")
sqs = boto3.client("sqs")

embeddings = OpenAIEmbeddings()


async def main():
    queue_url = "https://sqs.eu-central-1.amazonaws.com/246532218018/bubble-ingest"
    logger.info(f"running ingester for SQS: `{queue_url}`")

    if os.environ.get("DATABASE_URL") is None:
        connection_string = os.getenv("DATABASE_URL")
    else:
        connection_string = os.environ["DATABASE_URL"]

    logger.info(f"postgres connection: {connection_string}")

    # pool = await asyncpg.create_pool(connection_string)
    async with asyncpg.create_pool(connection_string) as pool:
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
                    content = json.loads(message["Body"])

                    # Get S3 object details from the S3 event
                    bucket = content["Records"][0]["s3"]["bucket"]["name"]
                    key = content["Records"][0]["s3"]["object"]["key"]

                    # LATER: optimize try block. AWS errors not handled
                    try:
                        document = load_document(bucket, key)
                        chunks = split_document(document)
                        texts = [chunk.page_content for chunk in chunks]
                        vectors = embeddings.embed_documents(texts)
                        zipped = zip(texts, vectors)

                        await persist(pool, zipped, document.metadata)
                        logger.info("Vector data persisted")

                    except ContentTypeException as ct:
                        logger.error(f"ContentType error: {ct}")
                    except ClientError as ce:
                        logger.error(f"AWS error: {ce}")
                    except exceptions.PostgresError as pe:
                        logger.error(f"postgres error: {pe}")
                    except exceptions.InterfaceError as ie:
                        logger.error(f"asyncpg error: {ie}")
                    except Exception as e:
                        logger.error(f"Generic error: {e}")
                    finally:
                        # Delete the message from the queue
                        sqs.delete_message(
                            QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"]
                        )
                        logger.info("SQS message deleted")

                        s3.delete_object(Bucket=bucket, Key=key)
                        logger.info(f"s3 file deleted")

                        logger.info(
                            "=========================================================================="
                        )


def load_document(bucket: str, key: str) -> Document:
    logger.info(f"Processing bucket {bucket}, key: {key}")

    # I have to load files manually with boto3,
    # just to be flexible with docuemnt loaders.
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/{key}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        s3.download_file(bucket, key, file_path)
        logger.info(f"File {key} has been downloaded successfully to {file_path}")

        s3metadata = get_s3metadata(bucket, key)
        elements = get_elements(file_path, s3metadata["content_type"])
        text = "\n\n".join([str(el) for el in elements])

        doc = Document(page_content=text, metadata=s3metadata)

        return doc


def get_elements(file_path, content_type) -> List[Element]:
    # not using auto, arguments: https://unstructured-io.github.io/unstructured/bricks.html
    logger.info(f"Processing {content_type}")
    if content_type == "application/pdf":
        # TODO: try again other `strategy` values, because some documents produce bad resuls, like:
        # /Users/kostik/Documents/neu_20200122_kuendigungs-_aenderungsantrag_digital.pdf
        return partition_pdf(file_path, strategy="fast")
    else:
        logger.error(f"Can't process document: unknown content_type `{content_type}`")
        # See if I need to use this for .docx file:
        # https://github.com/ankushshah89/python-docx2txt/
        raise ContentTypeException("Unknown content_type")


def get_s3metadata(bucket, key):
    # Head the object
    response = s3.head_object(Bucket=bucket, Key=key)

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
    logger.info(f"Chunks: {len(chunks)}")
    return chunks


async def persist(
    pool: asyncpg.Pool,
    zipped: List[Tuple[str, List[float]]],
    metadata: Dict[str, str],
):
    logger.info("Saving to postgres...")

    machine_id, full_path, content_type = map(
        clean_null_bytes,
        [metadata["machine_id"], metadata["full_path"], metadata["content_type"]],
    )

    async with pool.acquire() as conn:
        await register_vector(conn)
        async with conn.transaction():
            docs_query = """INSERT INTO public.documents(machine_id, full_path)
                            VALUES ($1, $2)
                            ON CONFLICT (machine_id, full_path)
                            DO UPDATE SET full_path = public.documents.full_path
                            RETURNING id;"""

            document_id = await conn.fetchval(docs_query, machine_id, full_path)
            logger.info(f"INSERT/UPDATE document_id: {document_id}")

            # First, delete old ones
            await conn.execute("DELETE FROM chunks WHERE document_id=$1;", document_id)

            emb_query = """INSERT INTO public.chunks(document_id, embedding, text)
            VALUES ($1, $2, $3);
            """

            # prepare & executemany for performance
            stmt = await conn.prepare(emb_query)

            entries = []

            for text, vector in zipped:
                text = clean_null_bytes(text)
                entries.append((document_id, vector, text))

            await stmt.executemany(entries)


def clean_null_bytes(field):
    """
    This function removes any null bytes from a string field. Otherwise we get an error from postgres
    """
    return field.replace("\0", "")


asyncio.get_event_loop().run_until_complete(main())

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class File(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.get("/", tags=["Root"])
async def read_root():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    chunks = text_splitter.split_text(
        "1us: Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well."
    )

    return {"message": chunks}


@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


# chunk and prepare with langchain
# save it to s3 (even before working with langchain?)
@app.put("/ingest/{document_hash}")
def ingest(document_hash: str, file: File):
    return {"file_name": file.name, "document_hash": document_hash}


# if s3 detects new files, it should make a beam.cloud call ("document_hash", "chunks" array)
# beam works with a webhook, saves complete data back to s3.
# new func here that checks for done embeddings on s3.

# TODO: pull embeddings from beam.cloud
# TODO: saves to chroma/other db

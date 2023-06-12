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
    return {"message": "Welcome to the API v.8"}


@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


# chunk and prepare with langchain
# send it to beam.cloud ("document_hash", "chunks" array)
@app.put("/ingest/{document_hash}")
def ingest(document_hash: str, file: File):
    return {"file_name": file.name, "document_hash": document_hash}


# TODO: pull embeddings from beam.cloud
# TODO: saves to chroma/other db

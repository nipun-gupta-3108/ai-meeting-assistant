from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def build_transcript_vector_store(
    transcript: str, collection_name: str = COLLECTION_NAME
) -> Chroma:
    print("Building Vector Store...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = create_embedding_model()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def create_transcript_retriever(vector_store: Chroma, k: int = 8):
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": 20}
    )

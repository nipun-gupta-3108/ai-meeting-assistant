from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def build_transcript_vector_store(
    transcript: str, collection_name: str = COLLECTION_NAME
) -> Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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


def load_transcript_vector_store() -> Chroma:
    embeddings = create_embedding_model()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def create_transcript_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

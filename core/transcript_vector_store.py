import logging
import os
import time

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# How old an untouched collection must be before the startup sweep will
# remove it. Kept well above any realistic single-session duration so an
# active session — in this process or another one running concurrently —
# is never at risk of being swept.
DEFAULT_STALE_COLLECTION_MAX_AGE_HOURS = float(
    os.getenv("STALE_COLLECTION_MAX_AGE_HOURS", "24")
)


@lru_cache(maxsize=1)
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def build_transcript_vector_store(
    transcript: str, collection_name: str = COLLECTION_NAME
) -> Chroma:
    logger.info("Building vector store (collection=%s)...", collection_name)

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
        # Stamped so a later, separate process can tell how old this
        # collection is without needing any persistence layer of our own
        # — see cleanup_stale_collections().
        collection_metadata={"created_at": str(time.time())},
    )

    return vector_store


def create_transcript_retriever(vector_store: Chroma, k: int = 8):
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": 20}
    )


def delete_collection(collection_name: str) -> None:
    """Delete a single Chroma collection once it's no longer needed (the
    user starts a new meeting, or a CLI session ends).

    Best-effort: a failed cleanup is logged but never raised — it should
    never block the user from starting a new meeting or crash an
    otherwise-successful session.
    """
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=create_embedding_model(),
            persist_directory=CHROMA_DIR,
        )
        store.delete_collection()
        logger.debug("Deleted Chroma collection: %s", collection_name)
    except Exception as exc:
        logger.warning(
            "Could not delete Chroma collection %s: %s", collection_name, exc
        )


def cleanup_stale_collections(
    max_age_hours: float = DEFAULT_STALE_COLLECTION_MAX_AGE_HOURS,
) -> None:
    """Remove Chroma collections that are old enough to be confidently
    considered abandoned, without touching anything recent.

    Design note: each pipeline run creates a uniquely-named collection
    (see core/pipeline.py) to keep one meeting's chunks from
    contaminating another's. delete_collection() removes a collection
    the moment its owning session ends normally, but a crashed process,
    a force-quit terminal, or a closed browser tab leaves its collection
    behind with no cleanup hook ever firing.

    An earlier version of this cleanup wiped the entire vector store
    directory on every process start. That's unsafe: if two processes
    (e.g. the CLI and the Streamlit app, or two Streamlit sessions) are
    running at the same time, a blanket wipe from one process's startup
    would delete the other's in-progress collection mid-session.

    This function avoids that by only removing collections whose
    recorded creation time (stamped into Chroma collection metadata at
    build time) is older than max_age_hours. Any collection belonging to
    a genuinely active session — in this process or a concurrent one —
    will always be far younger than the threshold, so it's never at
    risk. Only collections old enough to plausibly be orphaned get
    swept. This bounds disk growth to at most ~max_age_hours worth of
    crash-orphaned collections, rather than guaranteeing zero at all
    times — an acceptable tradeoff given there is no persistence layer
    yet to safely coordinate cleanup across processes.

    Best-effort throughout: never raises, so a failed sweep can never
    block the app from starting or crash a session.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception as exc:
        logger.warning(
            "Could not open Chroma client for stale-collection sweep: %s", exc
        )
        return

    try:
        collections = client.list_collections()
    except Exception as exc:
        logger.warning("Could not list Chroma collections: %s", exc)
        return

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for collection in collections:
        name = getattr(collection, "name", None)
        if not name:
            continue

        metadata = getattr(collection, "metadata", None) or {}
        try:
            created_at = float(metadata.get("created_at", 0))
        except (TypeError, ValueError):
            created_at = 0

        if created_at >= cutoff:
            # Recent (or missing an age we can trust as "old") — leave it.
            continue

        try:
            client.delete_collection(name)
            removed += 1
            logger.info("Removed stale Chroma collection: %s", name)
        except Exception as exc:
            logger.warning("Could not remove stale collection %s: %s", name, exc)

    if removed:
        logger.info("Stale-collection sweep removed %d collection(s).", removed)
    else:
        logger.debug("Stale-collection sweep found nothing to remove.")

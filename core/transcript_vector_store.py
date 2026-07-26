import logging
import os
import time
from functools import lru_cache

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_transcript_vector_store(
    transcript: str,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    logger.info(
        "Building vector store (collection=%s)...",
        collection_name,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = create_embedding_model()

    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=CHROMA_DIR,
            # Stamped so a later, separate process can tell how old this
            # collection is without needing any persistence layer of our own
            # — see cleanup_stale_collections().
            collection_metadata={
                "created_at": str(time.time()),
            },
        )

    except Exception:
        # Chroma may create the collection before all embeddings/documents
        # have been inserted. If construction then fails, the caller never
        # receives a vector-store object and therefore cannot clean that
        # partially-created collection itself.
        #
        # The collection name belongs to this pipeline invocation, so it is
        # safe to make a best-effort attempt to remove it here.
        logger.warning(
            "Vector-store construction failed for collection %s; attempting "
            "to remove any partial collection.",
            collection_name,
        )

        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)

            existing_names = {
                getattr(collection, "name", None)
                for collection in client.list_collections()
            }

            if collection_name in existing_names:
                client.delete_collection(collection_name)

                logger.info(
                    "Removed partial Chroma collection after build failure: %s",
                    collection_name,
                )

        except Exception as cleanup_exc:
            # Never mask the original vector-store construction error.
            # The normal stale-collection sweep remains the fallback.
            logger.warning(
                "Could not remove partial Chroma collection %s after build "
                "failure: %s",
                collection_name,
                cleanup_exc,
            )

        raise

    return vector_store


def create_transcript_retriever(
    vector_store: Chroma,
    k: int = 8,
):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
        },
    )


def delete_collection(collection_name: str) -> None:
    """Delete a single Chroma collection once it is no longer needed.

    Best-effort: a failed cleanup is logged but never raised. Cleanup should
    never block the user from starting a new meeting or crash an otherwise
    successful session.
    """
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=create_embedding_model(),
            persist_directory=CHROMA_DIR,
        )

        store.delete_collection()

        logger.debug(
            "Deleted Chroma collection: %s",
            collection_name,
        )

    except Exception as exc:
        logger.warning(
            "Could not delete Chroma collection %s: %s",
            collection_name,
            exc,
        )


def cleanup_stale_collections(
    max_age_hours: float = DEFAULT_STALE_COLLECTION_MAX_AGE_HOURS,
) -> None:
    """Remove Chroma collections old enough to be considered abandoned.

    Each pipeline run creates a uniquely named collection so one meeting's
    chunks cannot contaminate another meeting.

    delete_collection() removes a collection when its owning session ends
    normally. A crashed process, force-quit terminal, or closed browser tab
    can leave a collection behind without executing that cleanup.

    An earlier implementation deleted the entire vector-store directory on
    process startup. That was unsafe because another concurrently running
    process could still be using a collection in that directory.

    This implementation instead removes only collections whose recorded
    creation time is older than max_age_hours.

    Best-effort throughout: failures are logged but never raised.
    """

    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
        )

    except Exception as exc:
        logger.warning(
            "Could not open Chroma client for stale-collection sweep: %s",
            exc,
        )
        return

    try:
        collections = client.list_collections()

    except Exception as exc:
        logger.warning(
            "Could not list Chroma collections: %s",
            exc,
        )
        return

    cutoff = time.time() - (max_age_hours * 3600)

    removed = 0
    skipped_unrecognized = 0

    for collection in collections:
        name = getattr(
            collection,
            "name",
            None,
        )

        if not name:
            # Canary: chromadb has changed list_collections()'s return type
            # before (0.6.0-0.6.3 returned plain name strings instead of
            # Collection objects). If that behavior ever returns, silently
            # skipping every entry would disable cleanup indefinitely.
            skipped_unrecognized += 1
            continue

        metadata = (
            getattr(
                collection,
                "metadata",
                None,
            )
            or {}
        )

        try:
            created_at = float(
                metadata.get(
                    "created_at",
                    0,
                )
            )

        except (TypeError, ValueError):
            created_at = 0

        # Missing or invalid created_at metadata is intentionally treated as
        # stale. Such collections predate the timestamp-based lifecycle
        # mechanism (or have malformed metadata), so there is no trustworthy
        # evidence that they belong to a current session. Assigning 0 makes
        # them eligible for cleanup.
        if created_at >= cutoff:
            continue

        try:
            client.delete_collection(
                name,
            )

            removed += 1

            logger.info(
                "Removed stale Chroma collection: %s",
                name,
            )

        except Exception as exc:
            logger.warning(
                "Could not remove stale collection %s: %s",
                name,
                exc,
            )

    if skipped_unrecognized:
        logger.warning(
            "Stale-collection sweep found %d collection entry(ies) without a "
            "recognizable Collection object (no .name attribute) — the "
            "installed chromadb version may have changed list_collections()'s "
            "return type. The sweep is likely no longer able to detect stale "
            "collections; check the chromadb version pin in requirements.txt.",
            skipped_unrecognized,
        )

    if removed:
        logger.info(
            "Stale-collection sweep removed %d collection(s).",
            removed,
        )

    else:
        logger.debug(
            "Stale-collection sweep found nothing to remove.",
        )

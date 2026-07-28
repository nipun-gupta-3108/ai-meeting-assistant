from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from core.transcript_vector_store import (
    build_transcript_vector_store,
    create_transcript_retriever,
)
from core.llm_client import create_llm


def format_retrieved_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_transcript_rag_chain(transcript: str, collection_name: str):

    vector_store = build_transcript_vector_store(
        transcript, collection_name=collection_name
    )

    retriever = create_transcript_retriever(vector_store, k=4)

    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a meeting Q&A assistant. Use only the transcript context below to answer the user's question.

If the transcript does not contain the answer, reply exactly:
"I could not find this information in the meeting transcript."

Keep the answer brief, specific, and grounded in the transcript. When you quote or refer to a speaker's words, make that clear.

Transcript context:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    # Same prompt/LLM/parser as before — this sub-chain's semantics are
    # unchanged, only what wraps it has changed.
    answer_chain = prompt | llm | StrOutputParser()

    # Retrieval now happens exactly once per question, in this first stage.
    # Both branches below reuse that single retrieval result, so the
    # Documents used to build "context" for the LLM are the exact same
    # Documents later surfaced to the caller as sources — no second
    # retrieval is ever performed to produce citations.
    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            source_documents=retriever,
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_retrieved_documents(x["source_documents"])
            )
        )
        | RunnableParallel(
            answer=answer_chain,
            source_documents=RunnableLambda(lambda x: x["source_documents"]),
        )
    )

    return rag_chain


def _build_sources(source_documents) -> list:
    """Convert retrieved LangChain Documents into the small, framework-
    independent source structure returned to callers.

    Defensive by design: a Document with missing or non-integer
    "chunk_index" metadata is skipped rather than raising, matching the
    fallback philosophy already used in transcript_summary.py and
    transcript_insights.py. Duplicate chunk indices (e.g. if MMR ever
    returned the same chunk twice) are collapsed to a single entry.
    """
    sources = []
    seen_chunk_indices = set()

    for doc in source_documents or []:
        metadata = getattr(doc, "metadata", None) or {}
        chunk_index = metadata.get("chunk_index")

        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            continue

        if chunk_index in seen_chunk_indices:
            continue

        seen_chunk_indices.add(chunk_index)
        sources.append({"chunk_index": chunk_index})

    return sources


def format_sources_line(sources: list) -> str:
    """Render a list of {"chunk_index": int} source dicts as a single
    display line, e.g. "Sources: Transcript chunk 2, Transcript chunk 5".

    chunk_index is zero-based internally (see the "chunk_index" metadata
    set in core/transcript_vector_store.py). Displayed labels are
    one-based ("Transcript chunk 1" for chunk_index 0), since a zero-based
    label reads as a bug to a non-technical viewer rather than an
    intentional indexing choice. This is the single place that performs
    that zero-to-one-based conversion; callers should not re-derive it.

    Returns "" if there are no valid sources, so callers can skip
    rendering anything rather than showing an empty "Sources:" line.
    """
    labels = []

    for source in sources or []:
        if not isinstance(source, dict):
            continue

        chunk_index = source.get("chunk_index")
        if not isinstance(chunk_index, int):
            continue

        labels.append(f"Transcript chunk {chunk_index + 1}")

    if not labels:
        return ""

    return "Sources: " + ", ".join(labels)


def ask_transcript_question(rag_chain, question: str) -> dict:
    """Ask one question against a transcript's RAG chain.

    Returns a small, framework-independent contract:

        {
            "answer": <str>,
            "sources": [{"chunk_index": <int>}, ...],
        }

    "sources" always corresponds to the exact Documents used to generate
    "answer" for this call — no separate/second retrieval is performed to
    produce it. This function is intentionally stateless: it has no
    awareness of prior turns, matching current behavior exactly.
    """
    result = rag_chain.invoke(question)

    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    source_documents = (
        result.get("source_documents", []) if isinstance(result, dict) else []
    )

    return {
        "answer": answer,
        "sources": _build_sources(source_documents),
    }

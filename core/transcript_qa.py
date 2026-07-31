import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# Bounded conversational window: how many past chat messages (user +
# assistant turns combined) are converted into LangChain messages and fed
# into contextualization/answering. 6 messages = the last 3 user/assistant
# exchanges.
#
# Unbounded history would grow the prompt — and Groq token usage — with
# every turn, and would eventually push older, less-relevant turns ahead
# of the transcript context itself in the model's attention. A small fixed
# window keeps latency and cost predictable while still covering the
# common "why?" / "who else?" follow-up pattern this feature targets. 6 is
# a practical default, not derived from a specific benchmark on this
# project — worth revisiting if real usage shows follow-ups reaching
# further back than 3 exchanges.
MAX_HISTORY_MESSAGES = 6

CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question, \
rewrite the question, if needed, into a standalone question that can be understood \
without the chat history. Do NOT answer the question — only reformulate it. If the \
question is already standalone, return it unchanged. Return only the reformulated \
question, with no explanation, quotes, or extra text."""

ANSWER_SYSTEM_PROMPT = """You are an AI Meeting Assistant.

Answer the user's question using ONLY the transcript context provided below.

Instructions:
- Carefully read ALL transcript context before answering.
- The information needed to answer may be spread across multiple transcript excerpts. Combine information across all relevant excerpts into a single coherent answer.
- Do not rely on outside knowledge or make up facts.
- If the transcript contains only part of the answer, provide the partial answer and clearly mention what is missing.
- Only reply exactly:
"I could not find this information in the meeting transcript."
if the transcript context contains no relevant information at all.

Formatting:
- Be concise but complete.
- Use bullet points when listing action items, decisions, risks, participants, or questions.
- Preserve names, dates, deadlines, and technical terms exactly as they appear in the transcript.
- When quoting someone, make it clear that it is a quote.
- Do not mention the transcript chunks in the answer.

Transcript context:
{context}
"""


def format_retrieved_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def debug_documents(docs):
    print("\n" + "=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i}")
        print("-" * 80)
        print(doc.page_content)


def _build_contextualize_chain(llm):
    """Turns (chat_history, question) into a standalone question that no
    longer depends on prior turns — e.g. "Why Friday?" becomes something
    like "Why did Rahul choose Friday as the deadline?".

    This rewritten question is used ONLY to drive retrieval. The answer is
    still generated from the user's original, unmodified question (plus
    chat_history), so the assistant's reply reads naturally rather than
    echoing a robotic reformulation.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _resolve_standalone_question(inputs: dict, contextualize_chain) -> str:
    """Skip the extra LLM call entirely on a first turn (no chat_history),
    since there is nothing to contextualize against yet. This keeps
    single-turn Q&A exactly as fast/cheap as before this feature existed.
    """
    if not inputs.get("chat_history"):
        return inputs["question"]
    return contextualize_chain.invoke(inputs)


def build_transcript_rag_chain(transcript: str, collection_name: str):

    vector_store = build_transcript_vector_store(
        transcript, collection_name=collection_name
    )

    retriever = create_transcript_retriever(vector_store, k=12)

    llm = create_llm()
    contextualize_chain = _build_contextualize_chain(llm)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    answer_chain = answer_prompt | llm | StrOutputParser()

    # Retrieval now happens exactly once per question, using a standalone
    # (history-resolved) version of the question. Both the answer step and
    # the sources returned to the caller reuse this exact same retrieval
    # result — no second retrieval is ever performed to produce citations.
    rag_chain = (
        RunnableParallel(
            question=RunnableLambda(lambda x: x["question"]),
            chat_history=RunnableLambda(lambda x: x.get("chat_history", [])),
            standalone_question=RunnableLambda(
                lambda x: _resolve_standalone_question(x, contextualize_chain)
            ),
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(
                lambda x: (
                    debug_documents(docs := retriever.invoke(x["standalone_question"])),
                    docs,
                )[1]
            )
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


def ensure_rag_chain(result: dict) -> dict:
    """Lazily build and attach a conversational RAG chain to a meeting result.

    Live meetings (returned by core/pipeline.py) already have "rag_chain"
    and "collection_name" set at creation time, so this is a no-op for
    them — the guard below returns immediately.

    Historical meetings loaded via core/meeting_repository.get_meeting()
    have neither key, since SQLite never stores runtime objects (see that
    module's docstrings). The first time this is called for such a
    meeting, it builds a fresh vector store from the already-persisted
    transcript, wraps it in the exact same conversational RAG chain used
    for live meetings, and attaches both "rag_chain" and "collection_name"
    to `result` in place. Every later call for the same `result` object
    sees "rag_chain" already set and returns immediately without
    rebuilding anything — so this must be called on every question, but
    only does real work on the first one.

    The collection name uses a "history_" prefix (vs. pipeline.py's
    "meeting_" prefix) purely so the two lifecycles are easy to tell apart
    when inspecting logs or the Chroma directory — uniqueness itself comes
    from the uuid4 suffix and holds regardless of prefix.

    Callers remain responsible for eventually calling
    core.transcript_vector_store.delete_collection() on
    result["collection_name"] once the chain is no longer needed. The
    existing "Back to history" / "New meeting" cleanup path in
    streamlit_app.py already does this generically for any result that has
    a "collection_name", so it covers both live and (once built)
    historical meetings without modification.
    """
    if result.get("rag_chain") is not None:
        return result

    collection_name = f"history_{uuid.uuid4().hex}"
    result["rag_chain"] = build_transcript_rag_chain(
        result["transcript"], collection_name=collection_name
    )
    result["collection_name"] = collection_name

    return result


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


def _convert_chat_history_to_messages(chat_history: list) -> list:
    """Convert the app's stored chat-history dicts
    ({"role": "user"/"assistant", "content": str, ...}) into LangChain
    message objects, applying the bounded window (MAX_HISTORY_MESSAGES).

    Only "role" and "content" are read — extra keys such as the "sources"
    list Streamlit stores alongside assistant turns are ignored here, so
    callers can pass their stored dicts straight through unmodified.
    Entries with an unrecognized role are skipped rather than raising.
    """
    messages = []
    for turn in chat_history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages[-MAX_HISTORY_MESSAGES:]


def ask_transcript_question(
    rag_chain, question: str, chat_history: list | None = None
) -> dict:
    """Ask one question against a transcript's RAG chain.

    `chat_history`, if given, is the caller's own list of prior turns in
    the app's stored dict format (see _convert_chat_history_to_messages).
    It should NOT include the current `question` — only turns that came
    before it.

    Passing nothing (the default) reproduces the original single-turn
    behavior exactly: with no chat_history, _resolve_standalone_question
    skips contextualization entirely and retrieval runs on `question` as-is,
    identical to this function's behavior before conversational memory was
    added.

    Returns the same contract as before:

        {
            "answer": <str>,
            "sources": [{"chunk_index": <int>}, ...],
        }

    "sources" always corresponds to the exact Documents used to generate
    "answer" for this call — no separate/second retrieval is performed to
    produce it.
    """
    messages = _convert_chat_history_to_messages(chat_history)

    result = rag_chain.invoke({"question": question, "chat_history": messages})

    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    source_documents = (
        result.get("source_documents", []) if isinstance(result, dict) else []
    )

    return {
        "answer": answer,
        "sources": _build_sources(source_documents),
    }

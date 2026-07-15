import uuid

from utils.audio_preparation import prepare_audio_chunks
from core.audio_transcription import transcribe_audio_chunks
from core.transcript_summary import summarize_transcript, generate_meeting_title
from core.transcript_insights import (
    extract_meeting_insights_from_transcript,
)
from core.transcript_qa import build_transcript_rag_chain


def run_meeting_assistant_pipeline(source: str, language: str = "english") -> dict:
    print("Starting the AI meeting assistant...")

    chunks = prepare_audio_chunks(source)

    transcript = transcribe_audio_chunks(chunks, language)

    title = generate_meeting_title(transcript)

    summary = summarize_transcript(transcript)

    insights = extract_meeting_insights_from_transcript(transcript)

    action_item = insights["action_items"]
    decisions = insights["key_decisions"]
    questions = insights["open_questions"]

    # Unique per-run collection name prevents this meeting's transcript chunks
    # from being retrievable by, or contaminated with, any other meeting's chunks.
    collection_name = f"meeting_{uuid.uuid4().hex}"
    rag_chain = build_transcript_rag_chain(transcript, collection_name=collection_name)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

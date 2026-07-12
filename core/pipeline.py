from utils.audio_preparation import prepare_audio_chunks
from core.audio_transcription import transcribe_audio_chunks
from core.transcript_summary import summarize_transcript, generate_meeting_title
from core.transcript_insights import (
    extract_action_items_from_transcript,
    extract_key_decisions_from_transcript,
    extract_open_questions_from_transcript,
)
from core.transcript_qa import build_transcript_rag_chain


def run_meeting_assistant_pipeline(source: str, language: str = "english") -> dict:
    print("Starting the AI meeting assistant...")

    chunks = prepare_audio_chunks(source)

    transcript = transcribe_audio_chunks(chunks, language)
    print(f"Transcript preview (first 300 characters): {transcript[:300]}")

    title = generate_meeting_title(transcript)

    summary = summarize_transcript(transcript)

    action_item = extract_action_items_from_transcript(transcript)

    decisions = extract_key_decisions_from_transcript(transcript)
    questions = extract_open_questions_from_transcript(transcript)

    rag_chain = build_transcript_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }
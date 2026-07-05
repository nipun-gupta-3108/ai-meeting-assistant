from dotenv import load_dotenv
from utils.audio_preparation import prepare_audio_chunks
from core.audio_transcription import transcribe_audio_chunks
from core.transcript_summary import summarize_transcript, generate_meeting_title
from core.transcript_insights import (
    extract_action_items_from_transcript,
    extract_key_decisions_from_transcript,
    extract_open_questions_from_transcript,
)
from core.transcript_qa import build_transcript_rag_chain, ask_transcript_question

load_dotenv()


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


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_meeting_assistant_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Meeting Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\nAsk questions about this meeting transcript. Type 'exit' to quit.\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Session closed. Goodbye!")
            break
        if not question:
            continue
        answer = ask_transcript_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")






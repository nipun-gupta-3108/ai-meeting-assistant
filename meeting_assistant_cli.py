"""CLI entry point for the AI Meeting Assistant.

Run this directly to analyze a meeting from a YouTube URL or local audio /
video file, then chat with the resulting transcript.
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from core.logging_config import configure_logging

configure_logging()

from core.pipeline import run_meeting_assistant_pipeline
from core.transcript_qa import ask_transcript_question
from core.transcript_vector_store import delete_collection, cleanup_stale_collections

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Startup sweep: remove any old, plausibly-abandoned Chroma
    # collections (e.g. left behind by a crashed run) without touching
    # anything recent enough to belong to a concurrently running    # session. See core/transcript_vector_store.py:cleanup_stale_collections.
    cleanup_stale_collections()

    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    try:
        result = run_meeting_assistant_pipeline(source, language)
    except Exception as exc:
        # Full stack trace goes to the log for debugging; the user only
        # sees a short, actionable message on the terminal.
        logger.exception("Meeting analysis pipeline failed.")
        print(f"\nSomething went wrong while analyzing this meeting: {exc}")
        print("Check your input, API keys, and configuration, then try again.")
    else:
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

        try:
            while True:
                question = input("You: ").strip()
                if question.lower() in ["exit", "quit", "q"]:
                    print("Session closed. Goodbye!")
                    break
                if not question:
                    continue
                try:
                    answer = ask_transcript_question(rag_chain, question)
                except Exception as exc:
                    logger.exception("Q&A failed for question: %s", question)
                    print(f"\nSorry, I couldn't answer that: {exc}\n")
                    continue
                print(f"\nAssistant: {answer}\n")
        finally:
            # The RAG chain (and its Chroma collection) is about to go out
            # of scope for good — clean it up now rather than waiting for
            # the next process's startup sweep.
            collection_name = result.get("collection_name")
            if collection_name:
                delete_collection(collection_name)

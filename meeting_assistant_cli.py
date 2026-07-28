"""CLI entry point for the AI Meeting Assistant.

Run this directly to analyze a meeting from a YouTube URL or local audio /
video file, then chat with the resulting transcript.
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from core.logging_config import configure_logging

configure_logging()

from core.meeting_repository import initialize_database, save_meeting
from core.pipeline import run_meeting_assistant_pipeline
from core.transcript_qa import ask_transcript_question, format_sources_line
from core.transcript_vector_store import (
    cleanup_stale_collections,
    delete_collection,
)
from utils.audio_preparation import (
    DOWNLOAD_DIR,
    cleanup_stale_temp_files,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Startup sweeps:
    #
    # 1. Remove old, plausibly-abandoned Chroma collections left behind
    #    by crashed/terminated runs.
    #
    # 2. Remove old files from downloads/, which is an app-owned directory
    #    containing YouTube downloads and generated audio artifacts.
    #
    # We intentionally do NOT sweep arbitrary local-file directories used
    # by the CLI because those directories may contain user-owned files.
    cleanup_stale_collections()
    cleanup_stale_temp_files(DOWNLOAD_DIR)

    # Idempotent (CREATE TABLE IF NOT EXISTS) — safe to call on every run.
    initialize_database()

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
        # Persist the completed analysis exactly once, here on the
        # success path only — the except branch above never reaches this
        # line, so a failed pipeline run is never saved. A failure to save
        # is logged and printed as a warning but does not discard the
        # in-memory result: `result` is still used below for the printed
        # summary and the interactive Q&A session.
        try:
            meeting_id = save_meeting(result, language)
            result["meeting_id"] = meeting_id
        except Exception:
            logger.exception("Failed to save meeting to history database.")
            print("\n(Warning: could not save this meeting to history.)")
            result["meeting_id"] = None

        print("\n" + "=" * 60)
        print(f"Meeting Title: {result['title']}")
        print(f"\nSummary:\n{result['summary']}")
        print(f"\nAction Items:\n{result['action_items']}")
        print(f"\nKey Decisions:\n{result['key_decisions']}")
        print(f"\nOpen Questions:\n{result['open_questions']}")
        print("=" * 60)

        # Phase 2 — Chat with your meeting via RAG
        print(
            "\nAsk questions about this meeting transcript. " "Type 'exit' to quit.\n"
        )

        rag_chain = result["rag_chain"]

        # Conversational memory for this session only: a plain in-memory
        # list in the same {"role", "content"} dict shape the Streamlit app
        # stores, so both callers feed ask_transcript_question() identical
        # input. Windowing (last MAX_HISTORY_MESSAGES turns) is applied
        # inside ask_transcript_question, not here.
        chat_history = []

        try:
            while True:
                question = input("You: ").strip()

                if question.lower() in ["exit", "quit", "q"]:
                    print("Session closed. Goodbye!")
                    break

                if not question:
                    continue

                try:
                    qa_result = ask_transcript_question(
                        rag_chain,
                        question,
                        chat_history=chat_history,
                    )

                except Exception as exc:
                    logger.exception(
                        "Q&A failed for question: %s",
                        question,
                    )

                    print(f"\nSorry, I couldn't answer that: {exc}\n")

                    continue

                answer = qa_result["answer"]
                print(f"\nAssistant: {answer}")

                sources_line = format_sources_line(qa_result["sources"])
                if sources_line:
                    print(sources_line)

                print()

                # Only append after a successful turn, so a failed question
                # (caught above) never pollutes the history sent to
                # subsequent contextualization calls.
                chat_history.append({"role": "user", "content": question})
                chat_history.append({"role": "assistant", "content": answer})

        finally:
            # The RAG chain (and its Chroma collection) is about to go out
            # of scope for good — clean it up now rather than waiting for
            # the next process's startup sweep.
            collection_name = result.get("collection_name")

            if collection_name:
                delete_collection(collection_name)

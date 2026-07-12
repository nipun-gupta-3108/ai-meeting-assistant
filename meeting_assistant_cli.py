from dotenv import load_dotenv
from core.pipeline import run_meeting_assistant_pipeline
from core.transcript_qa import ask_transcript_question

load_dotenv()


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






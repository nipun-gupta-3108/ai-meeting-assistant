# Actionableitems, decision, questions
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import create_llm


def build_extraction_chain(system_prompt: str):
    llm = create_llm(temperature=0.2)
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items_from_transcript(transcript: str) -> str:
    chain = build_extraction_chain(
        "Review the meeting transcript and identify every concrete action item. "
        "For each item, include:\n"
        "- Task: what needs to be done\n"
        "- Owner: the responsible person, or 'Not specified'\n"
        "- Deadline: the due date or timing, or 'Not specified'\n\n"
        "Return a numbered list. If there are no action items, say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions_from_transcript(transcript: str) -> str:
    chain = build_extraction_chain(
        "Review the meeting transcript and list the important decisions that were made. "
        "Write each decision clearly as a numbered item. "
        "If no decisions were made, say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_open_questions_from_transcript(transcript: str) -> str:
    chain = build_extraction_chain(
        "Review the meeting transcript and list any unanswered questions, blockers, "
        "or topics that need follow-up. Write them as a numbered list. "
        "If nothing needs follow-up, say 'No open questions found.'"
    )
    return chain.invoke(transcript)

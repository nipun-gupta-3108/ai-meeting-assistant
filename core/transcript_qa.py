from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.transcript_vector_store import (
    build_transcript_vector_store,
    load_transcript_vector_store,
    create_transcript_retriever,
)
from core.llm_client import create_llm


def format_retrieved_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_transcript_rag_chain(transcript: str):

    vector_store = build_transcript_vector_store(transcript)

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

    # full LCEL Rag pipeline

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_retrieved_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_transcript_rag_chain():
    vector_store = load_transcript_vector_store()
    retriever = create_transcript_retriever(vector_store)

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

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_retrieved_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_transcript_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer

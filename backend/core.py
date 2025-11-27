from dotenv import load_dotenv
from typing import Any

load_dotenv()

from langsmith import Client
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


def run_llm(query: str, chat_history: list[dict[str, Any]] | None = None):
    if chat_history is None:
        chat_history = []
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    chat = ChatOllama(model="llama3.2", verbose=True, temperature=0)

    hub_client = Client()
    retrieval_qa_chat_prompt = hub_client.pull_prompt("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub_client.pull_prompt("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "answer": result["answer"],
        "source_documents": result["context"],
    }

    return new_result


if __name__ == "__main__":
    res = run_llm("What is LangChain?")
    print(res["answer"])

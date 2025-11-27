from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


def run_llm(query: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    chat = ChatOllama(model="llama3.2", verbose=True, temperature=0)

    hub_client = Client()
    retrieval_qa_chat_prompt = hub_client.pull_prompt("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), 
        combine_docs_chain=stuff_documents_chain
    )
    result= qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm("What is LangChain?")
    print(res["answer"])
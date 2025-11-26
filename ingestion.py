from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/v0.1")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="documentation-helper-index"
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()

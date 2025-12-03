from typing import Any
import os
import re

def run_llm(query: str, chat_history: list[dict[str, Any]] | None = None):
  """
  Analyzes the conflicting code blocks (HEAD vs Feature) and resolves the conflict.

  Args:
    query: The input string.
    chat_history: A list of dictionaries, where each dictionary represents a
      conversation history.  The keys of the dictionaries are the
      conversation turns, and the values are the corresponding
      responses.
  Returns:
    A list of raw code blocks, where each block is a dictionary with
    the following keys:
      - "query": The input string.
      - "answer": The output of the LLM.
      - "source_documents": A list of the source documents.
      - "context": The context of the conversation.
  """
  if chat_history is None:
    chat_history = []

  if query is None:
    return []

  try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    INDEX_NAME = "documentation-helper-index"
    
    def run_llm(query: str):
      if chat_history is None:
        chat_history = []
      
      embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
      docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
      
      return "What is LangChain?"
  except Exception as e:
    print(f"Error during LLM execution: {e}")
    return []
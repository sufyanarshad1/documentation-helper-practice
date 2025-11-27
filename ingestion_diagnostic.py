import asyncio
import os
import ssl
import time
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


print("\n🔧 Initializing embeddings model...")
start_time = time.time()
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)
print(f"⏱️  Embeddings init took: {time.time() - start_time:.2f}s")

print("\n🔧 Initializing ChromaDB...")
start_time = time.time()
vectorstore = Chroma(persist_directory="chroma_db_test", embedding_function=embeddings)
print(f"⏱️  ChromaDB init took: {time.time() - start_time:.2f}s")


def test_single_embedding():
    """Test how long a single embedding takes"""
    print("\n🧪 Testing single embedding generation...")
    test_doc = Document(
        page_content="This is a test document to measure embedding speed. " * 20,
        metadata={"source": "test"}
    )
    
    start_time = time.time()
    vectorstore.add_documents([test_doc])
    elapsed = time.time() - start_time
    print(f"⏱️  Single document (100 words) took: {elapsed:.2f}s")
    return elapsed


def test_batch_embedding(batch_size: int):
    """Test how long a batch of embeddings takes"""
    print(f"\n🧪 Testing batch of {batch_size} embeddings...")
    test_docs = [
        Document(
            page_content=f"This is test document {i}. " * 20,
            metadata={"source": f"test_{i}"}
        )
        for i in range(batch_size)
    ]
    
    start_time = time.time()
    vectorstore.add_documents(test_docs)
    elapsed = time.time() - start_time
    print(f"⏱️  Batch of {batch_size} documents took: {elapsed:.2f}s ({elapsed/batch_size:.2f}s per doc)")
    return elapsed


def test_parallel_batches():
    """Test parallel batch processing"""
    print("\n🧪 Testing parallel batch processing...")
    
    async def add_batch_async(batch_id: int, size: int):
        test_docs = [
            Document(
                page_content=f"Parallel test document {batch_id}-{i}. " * 20,
                metadata={"source": f"parallel_{batch_id}_{i}"}
            )
            for i in range(size)
        ]
        start = time.time()
        await asyncio.to_thread(vectorstore.add_documents, test_docs)
        elapsed = time.time() - start
        print(f"  ✓ Batch {batch_id} ({size} docs) took: {elapsed:.2f}s")
        return elapsed
    
    async def run_parallel():
        start_time = time.time()
        # Run 4 batches of 10 docs in parallel
        results = await asyncio.gather(*[add_batch_async(i, 10) for i in range(4)])
        total_elapsed = time.time() - start_time
        print(f"⏱️  4 parallel batches (40 docs total) took: {total_elapsed:.2f}s")
        print(f"    Average per batch: {sum(results)/len(results):.2f}s")
        print(f"    Speedup vs sequential: {sum(results)/total_elapsed:.1f}x")
        return total_elapsed
    
    return asyncio.run(run_parallel())


def check_ollama_performance():
    """Check if Ollama is responding quickly"""
    print("\n🧪 Testing Ollama API response time...")
    import requests
    
    start_time = time.time()
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        elapsed = time.time() - start_time
        print(f"⏱️  Ollama API response: {elapsed:.2f}s")
        if elapsed > 1:
            print("⚠️  WARNING: Ollama API is slow to respond!")
    except Exception as e:
        print(f"❌ Ollama API error: {e}")


if __name__ == "__main__":
    print("="*60)
    print("🔍 CHROMADB PERFORMANCE DIAGNOSTICS")
    print("="*60)
    
    check_ollama_performance()
    
    single_time = test_single_embedding()
    
    batch_10_time = test_batch_embedding(10)
    
    batch_50_time = test_batch_embedding(50)
    
    parallel_time = test_parallel_batches()
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Single doc:        {single_time:.2f}s")
    print(f"Batch of 10:       {batch_10_time:.2f}s ({batch_10_time/10:.2f}s per doc)")
    print(f"Batch of 50:       {batch_50_time:.2f}s ({batch_50_time/50:.2f}s per doc)")
    print(f"4 parallel x 10:   {parallel_time:.2f}s")
    print(f"\n💡 Estimated time for 1000 docs:")
    print(f"   Sequential (batches of 50): {(1000/50) * batch_50_time:.0f}s = {(1000/50) * batch_50_time/60:.1f} minutes")
    print(f"   Parallel (4x batches of 50): {(1000/50/4) * parallel_time:.0f}s = {(1000/50/4) * parallel_time/60:.1f} minutes")
    print("="*60)

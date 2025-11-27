import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# Use faster nomic-embed-text model (274MB, optimized for embeddings)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 5):
    """Process documents in batches asynchronously with maximum parallelism."""
    import time

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches - smaller batches for better parallelism
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Estimate time (nomic-embed-text is much faster ~2-3s per doc)
    estimated_time = (
        len(documents) * 3
    ) / 2  # ~3s per doc with faster model, 2 parallel workers
    log_info(
        f"⏱️  Estimated time: {estimated_time/60:.1f} minutes (with 2 parallel workers)"
    )

    start_time = time.time()
    completed = 0

    # Process all batches concurrently with true async using thread pool
    async def add_batch(batch: List[Document], batch_num: int):
        nonlocal completed
        try:
            batch_start = time.time()
            # Run blocking operation in thread pool for true parallelism
            await asyncio.to_thread(vectorstore.add_documents, batch)
            batch_time = time.time() - batch_start
            completed += len(batch)

            elapsed = time.time() - start_time
            docs_per_sec = completed / elapsed if elapsed > 0 else 0
            remaining = len(documents) - completed
            eta = remaining / docs_per_sec if docs_per_sec > 0 else 0

            log_success(
                f"VectorStore Indexing: Batch {batch_num}/{len(batches)} done in {batch_time:.1f}s | "
                f"Progress: {completed}/{len(documents)} ({completed/len(documents)*100:.1f}%) | "
                f"ETA: {eta/60:.1f}m"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently with controlled parallelism
    # Use 2 parallel workers to avoid overwhelming Ollama
    semaphore = asyncio.Semaphore(2)

    async def add_batch_with_limit(batch: List[Document], batch_num: int):
        async with semaphore:
            return await add_batch(batch, batch_num)

    tasks = [add_batch_with_limit(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches completed in {total_time/60:.1f} minutes! "
            f"({successful}/{len(batches)} batches, {len(documents)} documents)"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        # Skip documents with no content
        raw_content = tavily_crawl_result_item.get("raw_content")
        if not raw_content:
            log_warning(
                f"TavilyCrawl: Skipping {tavily_crawl_result_item['url']} - no content"
            )
            continue

        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 3000 chunk size and 300 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Info about fast nomic-embed-text model
    log_info(
        "✨ Using fast nomic-embed-text model (~3s per doc with parallelism)",
        Colors.GREEN,
    )

    # Process documents asynchronously with small batches for max parallelism
    await index_documents_async(splitted_docs, batch_size=10)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

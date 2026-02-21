import os
import time
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import CHROMA_DB_DIR

# Set up standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize these GLOBALLY so they don't reload on every single API call (Massive speed boost)
CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=Settings(anonymized_telemetry=False))
# BGE-M3 is powerful but slow to load; keeping it global saves 2-3 seconds per request
BGE_M3_EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")

def retrieve_context(user_query: str, n_results: int = 5):
    """
    Fetches relevant chunks from ChromaDB and returns them as a list of dictionaries.
    n_results: Defaulting to 5 to ensure good recall (finding the FAQ).
    """
    logger.info(f"Retrieving context for: '{user_query}'")
    
    try:
        collection = CHROMA_CLIENT.get_collection(name="clearpath_kb", embedding_function=BGE_M3_EF)

        # Retrieval
        results = collection.query(
            query_texts=[user_query],
            n_results=n_results
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_metadata = results['metadatas'][0]

        formatted_chunks = []
        logger.info("========== RETRIEVED CONTEXT CHUNKS ==========")
        
        for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadata)):
            # Ensure we capture metadata safely
            doc_name = meta.get("document_name") or meta.get("source") or "Unknown_Doc"
            
            chunk_dict = {
                "text": doc,
                "document": doc_name,
                "section": meta.get("section", "N/A"),
                "authority": meta.get("authority_score", 0.5)
            }
            formatted_chunks.append(chunk_dict)
            
            # Log for debugging
            logger.info(f"Chunk {idx + 1} | Source: {doc_name} | Words: {len(doc.split())}")
        
        logger.info("==============================================")
        return formatted_chunks

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

if __name__ == "__main__":
    # Terminal Test to verify FAQ document recall
    test_query = "can I use clearPath offline?"
    chunks = retrieve_context(test_query, n_results=5)
    
    print(f"\n[TEST RESULT] Retrieved {len(chunks)} chunks.")
    sources = [c['document'] for c in chunks]
    print(f"Sources found: {sources}")
    
    if any("FAQ" in s.upper() for s in sources):
        print("✅ SUCCESS: FAQ document detected in top 5 results.")
    else:
        print("⚠️ WARNING: FAQ not in top results. Consider re-indexing or checking 'n_results'.")
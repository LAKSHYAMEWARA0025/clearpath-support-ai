import re
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import CHROMA_DB_DIR
import logging

logger = logging.getLogger(__name__)

# 1. Rule-Based Domain Router
DOMAIN_RULES = {
    "Technical Documentation": ["api", "webhook", "architecture", "deployment", "json", "endpoint", "release"],
    "Customer Support Docs": ["error", "troubleshooting", "failed", "not working", "bug", "support", "ticket", "sla", "onboarding"],
    "Pricing & Plans": ["pricing", "cost", "plan", "upgrade", "enterprise", "free", "tier", "billing"],
    "Product Documentation": ["how to", "guide", "tutorial", "shortcut", "create", "export", "dashboard"]
}

def detect_domain(query: str):
    """Scans the query for keywords and returns the target domain."""
    query_lower = query.lower()
    for domain, keywords in DOMAIN_RULES.items():
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
                return domain
    return None

# 2. Hybrid Scoring Math
def calculate_hybrid_score(query: str, chunk_text: str, semantic_distance: float, authority_score: float):
    """
    Implements the assignment's exact hybrid ranking formula:
    0.7 semantic + 0.2 keyword + 0.1 authority
    """
    # 1. Semantic Score: ChromaDB returns cosine distance (0 to 2). 
    # We convert it to similarity (0 to 1).
    semantic_similarity = max(0.0, 1.0 - semantic_distance)
    
    # 2. Keyword Overlap Score
    query_words = set(re.findall(r'\w+', query.lower()))
    chunk_words = set(re.findall(r'\w+', chunk_text.lower()))
    if not query_words:
        keyword_score = 0.0
    else:
        # Ratio of query words found in the chunk
        overlap = query_words.intersection(chunk_words)
        keyword_score = len(overlap) / len(query_words)
        
    # 3. Final Weighted Math
    final_score = (0.7 * semantic_similarity) + (0.2 * keyword_score) + (0.1 * authority_score)
    
    return final_score, semantic_similarity

# 3. The Retrieval Engine
def retrieve_context(query: str, complexity_level: str = "simple"):
    """
    Executes the filtered vector search and applies hybrid ranking.
    Returns dynamic chunk counts based on complexity.
    """
    # Determine how many chunks to return dynamically
    chunk_limits = {"simple": 2, "medium": 4, "complex": 6}
    k = chunk_limits.get(complexity_level, 2)
    
    # We fetch extra chunks initially so we have room to re-rank them
    fetch_k = k + 4 
    
    # Connect to DB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=Settings(anonymized_telemetry=False))
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    collection = chroma_client.get_collection(name="clearpath_kb", embedding_function=sentence_transformer_ef)
    
    target_domain = detect_domain(query)
    search_params = {
        "query_texts": [query],
        "n_results": fetch_k
    }
    
    # Apply the strict metadata filter if a domain was detected
    if target_domain:
        search_params["where"] = {"domain": target_domain}
        logger.info(f"Retrieval routed strictly to domain: {target_domain}")
    else:
        logger.info("No specific domain detected, searching across all documents.")
        
    results = collection.query(**search_params)
    
    if not results['documents'][0]:
        return [], 0.0
        
    ranked_results = []
    total_semantic_similarity = 0.0
    
    # Loop through the results and apply our custom hybrid math
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        authority = float(metadata.get('authority_score', 0.5))
        
        final_score, semantic_sim = calculate_hybrid_score(query, text, distance, authority)
        
        ranked_results.append({
            "text": text,
            "document": metadata.get('document_name'),
            "section": metadata.get('section'),
            "domain": metadata.get('domain'),
            "final_score": final_score
        })
        
        total_semantic_similarity += semantic_sim
        
    # Sort by our custom hybrid score (highest to lowest)
    ranked_results = sorted(ranked_results, key=lambda x: x['final_score'], reverse=True)
    
    # Trim down to the requested dynamic chunk limit (2, 4, or 6)
    final_chunks = ranked_results[:k]
    
    # Calculate overall retrieval confidence based on semantic similarity of the chosen chunks
    confidence_score = (total_semantic_similarity / len(ranked_results)) if ranked_results else 0.0
    
    return final_chunks, confidence_score

# Simple test block to verify it works
if __name__ == "__main__":
    test_query = "Why is my API webhook failing?"
    chunks, confidence = retrieve_context(test_query, complexity_level="medium")
    print(f"\n--- Test Query: '{test_query}' ---")
    print(f"Retrieval Confidence: {confidence:.2f}")
    for idx, c in enumerate(chunks):
        print(f"\nResult {idx+1} (Score: {c['final_score']:.2f}):")
        print(f"Doc: {c['document']} | Section: {c['section']}")
        print(f"Text snippet: {c['text'][:100]}...")
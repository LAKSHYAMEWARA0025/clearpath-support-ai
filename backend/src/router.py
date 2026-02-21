import re
from config import MODEL_8B, MODEL_70B

# Markers that genuinely require high-reasoning capabilities
COMPLEX_INTENT_MARKERS = [
    "compare", "difference", "vs", "versus", "better", "pros and cons",
    "why", "reason", "cause", "explain the relationship",
    "step by step", "guide me", "how do i setup", "integrate", "troubleshoot"
]

def classify_and_route_query(query: str, retrieved_chunks: list):
    """
    Weighted Router: Decides model based on a combined score of 
    intent markers, query length, and document diversity.
    """
    query_lower = query.lower()
    score = 0
    reasons = []

    # 1. FIX FOR "TypeError": Ensure retrieved_chunks is handled safely
    unique_documents = set()
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            unique_documents.add(chunk.get('document') or chunk.get('source') or "Unknown")
        elif isinstance(chunk, str):
            source_match = re.search(r"Source:\s*([\w\.-]+)", chunk)
            unique_documents.add(source_match.group(1) if source_match else f"raw_{hash(chunk)}")

    # 2. EVALUATE INTENT (+3 points) - This is the strongest signal
    for marker in COMPLEX_INTENT_MARKERS:
        if re.search(rf'\b{re.escape(marker)}\b', query_lower):
            score += 3
            reasons.append(f"Intent: {marker}")
            break

    # 3. EVALUATE LENGTH (+1 point)
    word_count = len(query_lower.split())
    if word_count > 18:
        score += 1
        reasons.append("High word count")

    # 4. EVALUATE DOCUMENT DIVERSITY (+1 point)
    # Finding 2 docs is common; only 3+ docs suggests a complex synthesis.
    if len(unique_documents) >= 3:
        score += 1
        reasons.append(f"Diversity: {len(unique_documents)} docs")

    # --- DECISION LOGIC ---
    # Score >= 3: Route to Llama-3.3-70B
    # Score < 3: Route to Llama-3.1-8B
    
    routing_reason = ", ".join(reasons) if reasons else "Simple factual lookup"
    
    if score >= 3:
        return "complex", MODEL_70B, f"Escalated: {routing_reason} (Score: {score})"
    
    return "simple", MODEL_8B, f"Handled by 8B: {routing_reason} (Score: {score})"
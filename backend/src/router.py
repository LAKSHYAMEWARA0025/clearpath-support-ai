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
    Deterministic Rule-Based Router: Decides model based on score.
    Returns: (classification_label, model_string, routing_reason)
    """
    query_lower = query.lower()
    score = 0
    reasons = []

    # 1. Evaluate Intent (+3 points) - Rule: Specific complex keywords
    for marker in COMPLEX_INTENT_MARKERS:
        if re.search(rf'\b{re.escape(marker)}\b', query_lower):
            score += 3
            reasons.append(f"Intent detected: {marker}")
            break

    # 2. Evaluate Length (+1 point) - Rule: Long queries suggest complexity
    word_count = len(query_lower.split())
    if word_count > 18:
        score += 1
        reasons.append("High word count")

    # 3. Evaluate Document Diversity (+1 point) - Rule: Multiple sources need synthesis
    unique_docs = set()
    for chunk in retrieved_chunks:
        # Assuming metadata is attached to chunks; adjust key as per your retrieval.py
        source = chunk.get('metadata', {}).get('source', 'Unknown') if isinstance(chunk, dict) else "Unknown"
        unique_docs.add(source)
    
    if len(unique_docs) >= 3:
        score += 1
        reasons.append(f"Source diversity: {len(unique_docs)} files")

    # --- DECISION LOGIC ---
    routing_summary = ", ".join(reasons) if reasons else "Simple factual lookup"
    
    if score >= 3:
        # classification = "complex"
        return "complex", MODEL_70B, f"Escalated (Score {score}): {routing_summary}"
    
    # classification = "simple"
    return "simple", MODEL_8B, f"Basic (Score {score}): {routing_summary}"
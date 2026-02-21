import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def evaluate_output(query: str, llm_response: str, retrieved_chunks: List[Dict]) -> Tuple[bool, str]:
    """
    Evaluates the LLM output for reliability before showing it to the user.
    """
    query_lower = query.lower()
    response_lower = llm_response.lower()
    
    # 1. Hallucination Guard: Context vs. Content
    # If retrieval found nothing but the model is "confidently" yapping
    if not retrieved_chunks:
        safety_keywords = ["do not have", "don't have", "not enough information", "unable to find"]
        if not any(kw in response_lower for kw in safety_keywords):
            return True, "Hallucination: Answered without context"

    # 2. Semantic Relevancy Check (Simple keyword-overlap baseline)
    # Checks if the response actually contains key nouns from the query
    # important_keywords = [word for word in query_lower.split() if len(word) > 4]
    # matches = [word for word in important_keywords if word in response_lower]
    
    # # If the user asks about "ClearPath API" and neither word is in the answer...
    # if important_keywords and len(matches) / len(important_keywords) < 0.2:
    #     return True, "Relevancy: Response does not address query keywords"
    # 2. Semantic Relevancy Check (Improved)
    # Filter out common question words and short filler words
    stop_words = {"what", "is", "the", "does", "how", "can", "you", "about"}
    query_words = [re.sub(r'[^\w]', '', w) for w in query_lower.split() if w not in stop_words]
    important_keywords = [w for w in query_words if len(w) > 3]
    
    if important_keywords:
        matches = [word for word in important_keywords if word in response_lower]
        match_ratio = len(matches) / len(important_keywords)
        
        # LOGIC: 
        # For very short queries (1-2 keywords), we need at least one match.
        # For longer queries, we need at least 25% match ratio.
        if len(important_keywords) <= 2:
            is_relevant = len(matches) >= 1
        else:
            is_relevant = match_ratio >= 0.25

        if not is_relevant:
            return True, f"Relevancy: Only matched {len(matches)}/{len(important_keywords)} keywords"

    # 3. Procedural Integrity (The "Support UX" Rule)
    # If the query is a "How-to", the response MUST be structured
    procedural_intent = any(kw in query_lower for kw in ["how", "step", "guide", "process", "setup"])
    if procedural_intent:
        # Check for Markdown lists (1., -, *, or even 'Step 1')
        has_list = bool(re.search(r'(?m)^(\d+\.|\-|\*|Step\s\d)\s', llm_response))
        if not has_list and len(llm_response.split()) > 40: # Only flag long unstructured blocks
            return True, "Structure: Procedural answer missing lists/steps"

    # 4. Negative Constraint Check
    # Ensure the model didn't leak system prompt secrets or "Untrusted" tags
    # leakage_markers = ["<context>", "untrusted", "system prompt", "internal instructions"]
    # if any(marker in response_lower for marker in leakage_markers):
    #     return True, "Security: Internal tag leakage detected"
    # 4. Negative Constraint Check (Updated)
    leakage_markers = ["<context>", "untrusted", ".pdf", ".docx", ".txt", "document:"]
    if any(marker in response_lower for marker in leakage_markers):
        return True, "Security: Internal metadata leakage detected"
    # 5. Success
    logger.info("Evaluator: Response passed all guardrails.")
    return False, "Passed"
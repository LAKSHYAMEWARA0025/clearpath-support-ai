import re
import logging

logger = logging.getLogger(__name__)

def evaluate_output(query: str, llm_response: str, retrieved_chunks: list):
    """
    Evaluates the LLM output for reliability before showing it to the user.
    Returns:
        tuple: (is_flagged: bool, flag_reason: str)
    """
    query_lower = query.lower()
    response_lower = llm_response.lower()
    
    # 1. No-Context Hallucination Detection
    # If the retrieval system found NOTHING, but the LLM still tried to answer instead
    # of using the required safety phrase, it is hallucinating.
    if len(retrieved_chunks) == 0:
        if "i do not have enough information" not in response_lower:
            logger.warning("Evaluator Flag: No-context hallucination detected.")
            return True, "No-context hallucination"

    # 2. Refusal / Non-Answer Detection
    # If the model explicitly states it cannot answer the question based on the docs
    refusal_phrases = [
        "i do not have enough information",
        "i cannot help",
        "contact support",
        "i don't know",
        "i am unable to answer"
    ]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            logger.warning(f"Evaluator Flag: Refusal detected ('{phrase}')")
            return True, "Refusal or non-answer"

    # 3. Custom Domain-Specific Check: Missing Structured Steps
    # Support UX rule: Procedural questions MUST have lists or steps.
    procedural_keywords = ["how to", "how do i", "steps to", "guide", "setup", "configure"]
    is_procedural = any(kw in query_lower for kw in procedural_keywords)
    
    if is_procedural:
        # Regex looks for "1.", "-", or "*" at the start of a line indicating a list
        has_structure = bool(re.search(r'(?m)^(\d+\.|\-|\*)\s', llm_response))
        if not has_structure:
            logger.warning("Evaluator Flag: Missing structured steps for procedural query.")
            return True, "Missing structured steps"

    # 4. Length / Completeness Check (Based on your suggestion)
    # If the query is highly complex/long, but the answer is suspiciously brief
    if len(query_lower.split()) > 15 and len(response_lower.split()) < 15:
        logger.warning("Evaluator Flag: Response suspiciously short for complex query.")
        return True, "Incomplete response"

    # If it passes all checks, it is safe to show to the user normally
    logger.info("Evaluator: Response passed all checks.")
    return False, "Passed"

# --- Test Block ---
if __name__ == "__main__":
    print("\n--- Testing Evaluator ---\n")
    
    # Test 1: Hallucination
    flag, reason = evaluate_output("What is ClearPath?", "ClearPath is a SaaS tool.", [])
    print(f"Test 1 (Hallucination): Flagged? {flag} | Reason: {reason}")
    
    # Test 2: Refusal
    flag, reason = evaluate_output("What is the secret admin password?", "I do not have enough information to answer that question.", [{"text": "some context"}])
    print(f"Test 2 (Refusal): Flagged? {flag} | Reason: {reason}")
    
    # Test 3: Missing Structure (Custom Check)
    flag, reason = evaluate_output("How do I export data?", "You can export data by going to settings and clicking the export button at the bottom of the screen.", [{"text": "some context"}])
    print(f"Test 3 (Bad Structure): Flagged? {flag} | Reason: {reason}")
    
    # Test 4: Good Response
    good_response = "Here is how to export:\n1. Go to settings\n2. Click export"
    flag, reason = evaluate_output("How do I export data?", good_response, [{"text": "some context"}])
    print(f"Test 4 (Good Structure): Flagged? {flag} | Reason: {reason}\n")
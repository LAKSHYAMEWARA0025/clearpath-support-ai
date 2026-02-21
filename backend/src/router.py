import re
from config import MODEL_8B, MODEL_70B

# 1. Define distinct domain keywords for Cross-Domain Detection
DOMAIN_RULES = {
    "Technical": ["api", "webhook", "architecture", "deployment", "json", "endpoint"],
    "Support": ["error", "troubleshooting", "failed", "not working", "bug", "ticket", "sla"],
    "Pricing": ["pricing", "cost", "plan", "upgrade", "enterprise", "free", "tier", "billing"],
    "Product": ["how to", "guide", "tutorial", "shortcut", "dashboard", "export"]
}

# 2. Define strict reasoning markers
REASONING_MARKERS = [
    "why", "reason", "cause", "how come",  # Causal
    "vs", "difference", "compare", "better", "instead",  # Comparison
    "after", "before", "then", "step", "first", "next"  # Sequential
]

def count_domains_triggered(query_lower: str) -> int:
    """Helper function to count how many distinct domains a query spans."""
    triggered_domains = 0
    for domain, keywords in DOMAIN_RULES.items():
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', query_lower):
                triggered_domains += 1
                break  # Once a domain is triggered, stop checking its other keywords
    return triggered_domains

def classify_and_route_query(query: str):
    """
    Deterministic, rule-based router to select the appropriate LLM.
    Returns: (classification, model_string, trigger_reason)
    """
    query_lower = query.lower()
    
    # Rule 1: Query Length (> 20 words implies heavy context/background info)
    word_count = len(query_lower.split())
    if word_count > 20:
        return "complex", MODEL_70B, "Length > 20 words"
        
    # Rule 2: Multiple Questions
    if query.count("?") > 1:
        return "complex", MODEL_70B, "Multiple questions detected"
        
    # Rule 3: Cross-Domain Detection (Requires merging distinct knowledge bases)
    domains_triggered = count_domains_triggered(query_lower)
    if domains_triggered >= 2:
        return "complex", MODEL_70B, f"Cross-domain query ({domains_triggered} domains)"
        
    # Rule 4: Reasoning & Troubleshooting Markers
    for marker in REASONING_MARKERS:
        if re.search(rf'\b{re.escape(marker)}\b', query_lower):
            return "complex", MODEL_70B, f"Reasoning marker '{marker}'"
            
    # Explicitly check for Support/Troubleshooting words as they always need reasoning
    for kw in DOMAIN_RULES["Support"]:
        if re.search(rf'\b{re.escape(kw)}\b', query_lower):
            return "complex", MODEL_70B, f"Troubleshooting keyword '{kw}'"

    # Fallback: Simple lookup (e.g., greetings, single facts)
    return "simple", MODEL_8B, "Default fallback (Simple fact lookup)"

# --- Test Block ---
if __name__ == "__main__":
    test_queries = [
        "What is ClearPath?", 
        "How much is the enterprise plan?",
        "What is the difference between Pro and Enterprise?", # Should trigger Comparison
        "My API webhook is throwing an error.", # Should trigger Support/Troubleshooting
        "How much does the API cost?", # Should trigger Cross-Domain (Tech + Pricing)
        "I am trying to set up a new project for my team of 30 people and I need to know if the billing cycle is monthly or annually before I commit?", # Should trigger Length
        "What is the SLA? And how do I export my dashboard?" # Should trigger Multiple Questions
    ]
    
    print("\n--- Testing Deterministic Router ---\n")
    for q in test_queries:
        classification, model, reason = classify_and_route_query(q)
        print(f"Query: '{q}'")
        print(f"Route: [{classification.upper()}] -> {model}")
        print(f"Why? : {reason}\n")
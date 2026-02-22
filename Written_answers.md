# 📝 ClearPath — Written Answers

## AI Usage
> **Note:** As per assignment requirements, if you used an LLM to assist in code generation or debugging, list the exact prompts used in this section.

* **Project Orchestration:** "Here is my current orchestrator: Query → Router → Retriever → LLM → Evaluator → Logger. Is this correct?"
* **Architectural Guidance:** "Should routing happen before retrieval, or should retrieval inform routing? Explain which is architecturally superior and why."
* **Security & Prompt Injection:** "Explain how to design prompts so that retrieved documents are treated as data, not executable instructions."
* **Cost Analysis:** "If this system handles 5,000 queries per day using two Groq models, how should I estimate token usage? Where is the biggest cost driver? What is the highest-ROI optimization?"

---

## Q1 — Routing Logic

For the routing logic, we established a set of predefined rules that act as high-intent markers—specifically looking for keywords like "comparison," "versus," "reason," or "troubleshoot." If these terms are detected, the system automatically routes the query to the **Llama-3.3-70B** model to leverage its superior reasoning capabilities. 

We also assign scores based on query length and the presence of multiple questions within a single prompt; if the cumulative score exceeds our threshold, the complex model is triggered, otherwise, it defaults to the simple model. We drew the boundary here because this weighted system is incredibly robust and flexible, allowing us to easily adjust the "importance" of any rule—like increasing the weight of query length—to meet changing user needs. 

A perfect example of a misclassification was the query, **"Is the mobile app better than the web dashboard?"** Even though this is a comparison, it bypassed the complex route because the word "better" wasn't in our keyword list and the query was too short to hit the length threshold. To improve this without adding LLM latency, I would look at "Source Diversity"—if a query pulls chunks from several different documents, it’s a clear signal that the 70B model is needed to synthesize that information, even if specific keywords are missing.

---

## Q2 — Retrieval Failures

A notable failure occurred when I asked, **"Is ClearPath available offline?"** This query should have been a straightforward hit on the FAQ document, but the system failed to retrieve it because the relevant text used the pronoun **"It"** instead of the brand name **"ClearPath,"** causing the vector search to lose the necessary context. 

Furthermore, my retrieval logic mistakenly categorized this as a technical integration issue rather than a general customer support query, leading the system to prioritize troubleshooting docs over the actual FAQ folder. To fix this, I would enhance the retrieval logic by expanding the classification keywords and increasing our "Top-K" retrieval threshold; currently, I only pull 5 chunks, but the correct answer might have been sitting at rank #6 or #7, just out of reach. By widening that net and improving how we classify intent, we can ensure the system doesn't overlook the right document just because of a pronoun or a narrow search window.

---

## Q3 — Cost and Scale

### Daily Token Usage Estimation (5,000 queries/day)

If we assume a typical workday for ClearPath, about 80% of questions are going to be quick "Where is this feature?" types, while the other 20% will be deeper "How do I fix this workflow?" headaches.

* **The Setup:** For every query, we aren't just sending the user's question; we’re also sending several paragraphs of documentation (the "Context") so the AI knows what it's talking about.
    * **Input:** ~1,000 tokens (Question + Docs)
    * **Output:** ~200 tokens (The Answer)
    * **Total:** 1,200 tokens per query.

* **The Math:**
    * **Llama 3.1 8B (Simple - 4,000 queries):** $4,000 \times 1,200 = 4.8 \text{ million tokens/day}.$
    * **Llama 3.3 70B (Complex - 1,000 queries):** $1,000 \times 1,200 = 1.2 \text{ million tokens/day}.$
* **Total Daily Volume:** **6 million tokens.**

### Cost Analysis

**Where is the biggest cost driver?**
The biggest "wallet-drainer" isn't actually the AI’s answer—it’s the **Input Tokens**. In RAG, we are basically "over-packing" the prompt with documentation chunks just to make sure the AI has enough info. Since the input is usually 5x to 10x larger than the actual answer, we’re paying mostly for the "reading material" we give the AI, not the "writing" it does.

**The Highest-ROI Change: Smart Pruning**
The best way to save money without hurting the quality is **Reranking**. Right now, we’re throwing 5 chunks of text at the AI "just in case." With a reranker, we could pull 20 chunks initially using a cheap search, use a tiny, lightning-fast model to pick the *best* 2, and only send those to the expensive 70B model. It’s like proofreading the AI's "reading list" so it only reads what's absolutely necessary.

**What optimization would I avoid?**
I would absolutely avoid **capping the answer length**. It’s tempting to tell the AI "keep it under 50 words" to save a few cents, but that’s a trap. In support, a short answer that misses a crucial step is useless. I’d rather spend a bit more on a complete, helpful answer than save money by giving the user a half-baked response that forces them to ask again anyway.

---

## Q4 — What Is Broken

### The Significant Flaw: "Tunnel Vision" (Lack of Global Context)
The most honest limitation of this system is that it suffers from **retrieval tunnel vision**. Our current RAG setup is great at "needle-in-a-haystack" questions—finding a specific fact about a password reset, for example. However, it completely fails at "the shape of the haystack" questions.

If a user asks, *"What is Clearpath’s general philosophy on user privacy across all modules?"*, the system grabs 3 or 4 disconnected snippets of text. It’s like trying to understand a 30-chapter book by only reading five random sentences. The LLM never sees the "big picture" because it only receives tiny, isolated fragments of data, making it impossible to perform true cross-document reasoning or high-level summarization.

### Why I Shipped With It Anyway
In a real-world deployment, you’d solve this with a Knowledge Graph or a complex "Map-Reduce" approach where you summarize documents before searching them. However, these architectures are massive undertakings. Given the constraints of a take-home assignment—and the specific rule against using "managed" RAG-as-a-service tools—I chose to prioritize a **rock-solid, transparent baseline**.

If I had more time I'd have gone for parent-Document Retrieval.

# 🚀 ClearPath-RAG: Hybrid Intelligent Support Bot

**ClearPath-RAG** is a high-performance Retrieval-Augmented Generation (RAG) system designed for project management support. It features a **Deterministic Hybrid Router** that intelligently toggles between **Llama-8B** and **Llama-70B** models to balance sub-second latency with deep reasoning capabilities.

---

## 🛠️ Local Setup & Commands

### 1. Backend Setup (FastAPI)
```
# Navigate to backend
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your Groq API Key
echo "GROQ_API_KEY=your_key_here" > .env

# Start the server
python src/main.py

Note: The backend will be live at http://localhost:8000.
```
Frontend Setup (React)
```
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
Note: The dashboard will be live at http://localhost:5173.
```
 ## 🧠 AI Architecture & Environment

### Models Used (via Groq)
We utilize the **Groq Llama-3 API** for lightning-fast inference, employing a tiered strategy to optimize for both cost and reasoning depth:

* **Simple Route:** `llama-3.1-8b-instant`
    * *Usage:* Direct factual lookups, greetings, and single-document queries.
    * *Benefit:* Optimized for sub-second latency and high efficiency.
* **Complex Route:** `llama-3.3-70b-versatile`
    * *Usage:* Multi-document synthesis, complex comparisons, troubleshooting, and multi-step reasoning.
    * *Benefit:* High parameter count for deep contextual understanding across the 30-doc knowledge base.

---

## 🛠️ Routing Logic

Our system implements a **Deterministic Weighted Scoring Algorithm** to classify user intent before choosing an inference path. This ensures model selection is transparent and repeatable.

### Scoring Metrics
The router evaluates every incoming query based on three primary signals:

1.  **Keyword Match (+3 points)**
    * Looks for high-reasoning markers such as: `compare`, `difference`, `vs`, `setup`, `how-to`, or `troubleshoot`.
2.  **Query Length (+1 point)**
    * Triggered when a query exceeds **18 words**, as longer queries typically involve more nuance or multiple questions.
3.  **Source Density (+1 point)**
    * Triggered when the retrieval engine identifies relevant information across **3 or more distinct PDF sources**, indicating a need for synthesis.

### Decision Threshold
The final routing decision is governed by the following mathematical condition:

$$\text{Total Score} \ge 3 \implies \text{Llama-3.3-70b}$$
$$\text{Total Score} < 3 \implies \text{Llama-3.1-8b}$$



> **Note:** Because a keyword match provides +3 points instantly, any query containing high-intent terms is immediately promoted to the 70B model, while simple factual lookups default to the 8B model for maximum cost-efficiency.

## 🌟 Features & Bonus Challenges

### 1. Real-Time Token & Latency Metering
The UI features a **Live Trace Sidebar** that displays JSON logs for every interaction, providing deep visibility into the system's performance:
* **$tokens_{in}$** and **$tokens_{out}$**: Directly captured from the Groq API response.
* **Total Latency (ms)**: Measured from the initial request to the final response.
* **Model Classification**: Displays whether the query was routed to the `simple` or `complex` path.

### 2. Evaluator Guardrails (Self-Correction)
Every response passes through a dedicated evaluation layer before reaching the user to ensure reliability:
* **Refusal Detection**: Automatically flags responses where the LLM indicates it cannot answer or lacks information.
* **Context Grounding**: Verifies if the answer is strictly derived from the provided 30 documentation PDFs.
* **UX Warning**: If a "Low Confidence" flag is triggered by the evaluator, the UI prepends a clear notice:
    > ⚠️ **Low confidence — please verify with support.**

### 3. Frontend Complexity Mapping
To ensure transparency in the routing decision, the UI dynamically maps the `model_used` key to a human-readable **Complexity** label:
* If the model is **8b**, it is labeled as **"Simple"**.
* Otherwise, it is labeled as **"Complex"**.

This provides immediate insight into how the **Deterministic Router** categorized the user's intent.
## Project Structure

The project is organized into a clear separation of concerns between the **FastAPI** backend and the **React** frontend. Below is the directory map:

```plaintext
clearpath-support-ai
├── backend/
│   ├── src/
│   │   ├── main.py          # FastAPI Endpoints & API Logic
│   │   ├── retrieval.py     # RAG Pipeline: ChromaDB & PDF Vectorization
│   │   ├── router.py        # Layer 2: Deterministic Rule-Based Logic
│   │   ├── llm.py           # Layer 1: Groq API & Prompt Engineering
│   │   └── evaluator.py     # Layer 3: Self-Correction & Output Validation
│   └── docs/                # Knowledge Base: 30 ClearPath Support PDFs
│
└── frontend/
    └── src/
        ├── App.tsx          # Main Chat Interface & Live Trace Logic
        └── App.css          # UI Styling & Terminal-style Log Formatting
```

## ⚠️ Known Issues & Limitations

While the system is robust for its current scope, users should be aware of the following technical constraints and expected behaviors:

### 1. Initial Indexing Latency
* **Behavior**: Upon the very first launch, the system must process and vectorize the 30 PDF documents into the **ChromaDB** vector store.
* **Impact**: This one-time process may take **10-15 seconds**. Subsequent starts are near-instant as the database is persisted locally.

### 2. ChromaDB Telemetry Warnings
* **Behavior**: You may observe a `chromadb.telemetry` or anonymous collection warning in your terminal logs.
* **Resolution**: This is a standard notification from the Chroma library and **does not affect RAG performance**, data privacy, or the accuracy of the bot's responses.

### 3. API Rate Limiting (Error 429)
* **Behavior**: The **Groq Free Tier API** has strict rate limits on requests per minute (RPM) and tokens per minute (TPM).
* **Impact**: Under heavy concurrent usage or rapid-fire querying, the backend may return a `429: Rate Limit Reached` error. If this occurs, wait 60 seconds before retrying.



---

### 🛡️ System Recommendations
* **PDF Formatting**: Ensure all documents in the `/docs` folder are text-searchable (not scanned images) for optimal retrieval.
* **Environment**: For best results, run the backend in a dedicated virtual environment as specified in the **Setup** section to avoid library version conflicts.

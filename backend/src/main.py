import os
import logging
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. SILENCE ALL NOISY LOGS ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger('chromadb.telemetry.posthog').disabled = True
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 2. INTERNAL MODULE IMPORTS ---
from retrieval import retrieve_context
from router import classify_and_route_query
from llm import generate_response
from evaluator import evaluate_output
from logger import router_log 

app = FastAPI(title="ClearPath AI Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. STARTUP LOG ---
@app.on_event("startup")
async def startup_event():
    # Clear terminal for a professional look
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*50)
    print("🚀 CLEARPATH AI CORE: ONLINE")
    print("📡 ENDPOINT: http://localhost:8000/api/chat")
    print("🛡️ STATUS: Ready for Frontend Queries")
    print("="*50 + "\n")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    confidence_flag: bool
    flag_reason: str
    classification: str 
    model_used: str
    tokens_input: int 
    tokens_output: int
    latency_ms: int

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    query = request.query
    
    # 1. Layer 1: Retrieval
    retrieved_chunks = retrieve_context(query)
    
    # 2. Layer 2: Routing 
    # classification returns "simple" or "complex" based on your score logic
    classification, model_name, route_reason = classify_and_route_query(query, retrieved_chunks)
    
    # 3. Layer 3: LLM Generation
    llm_answer, tokens_input, tokens_output = generate_response(query, retrieved_chunks, model_name)
    
    # 4. Layer 4: Evaluation
    # is_flagged is boolean, flag_reason is the status (e.g., "Passed", "Refusal", "No Context")
    is_flagged, flag_reason = evaluate_output(query, llm_answer, retrieved_chunks)
    
    # --- Refined UX Logic ---
    final_answer = llm_answer
    # We flag the response if the evaluator found a critical issue
    critical_risks = ["Hallucination", "Security", "Relevancy", "Structure", "Refusal", "No Context"]
    is_critical_flag = is_flagged and any(k in flag_reason for k in critical_risks)
    
    if is_critical_flag:
        # User-facing message as per assignment requirement
        final_answer = f"⚠️ **Low confidence — please verify with support.**\n\n{llm_answer}"
        
    latency_ms = round((time.time() - start_time) * 1000)
    
    # Final structured log for the terminal/backend logger
    router_log.info({
        "query": query, 
        "classification": classification,
        "model": model_name,
        "tokens_in": tokens_input,
        "tokens_out": tokens_output,
        "latency": latency_ms
    })
    
    return ChatResponse(
        answer=final_answer,
        confidence_flag=is_critical_flag,
        flag_reason=flag_reason,        # e.g., "Passed"
        classification=classification,  # e.g., "complex"
        model_used=model_name,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        latency_ms=latency_ms
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
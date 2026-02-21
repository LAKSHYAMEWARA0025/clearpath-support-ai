import os
import logging

# --- 1. MUTE NOISY LIBRARIES (Must be at the absolute top) ---
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import time
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    confidence_flag: bool
    flag_reason: str
    model_used: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    query = request.query
    
    # 1. Layer 1: Retrieval
    retrieved_chunks = retrieve_context(query)
    
    # 2. Layer 2: Routing
    classification, model_name, route_reason = classify_and_route_query(query, retrieved_chunks)
    
    # 3. Layer 3: LLM Generation
    llm_answer, tokens_input, tokens_output = generate_response(query, retrieved_chunks, model_name)
    
    # 4. Layer 4: Evaluation
    is_flagged, flag_reason = evaluate_output(query, llm_answer, retrieved_chunks)
    
    # --- Refined UX Logic: Filter Warnings ---
    final_answer = llm_answer
    # Only show warning for critical accuracy risks (not for honest refusals)
    critical_risks = ["Hallucination", "Security", "Relevancy", "Structure"]
    is_critical_flag = is_flagged and any(k in flag_reason for k in critical_risks)
    
    if is_critical_flag:
        final_answer = f"⚠️ **Notice:** This response is auto-generated and may require verification.\n\n{llm_answer}"
        
    latency_ms = round((time.time() - start_time) * 1000)
    
    router_log.info({
        "query": query, 
        "model": model_name,
        "flagged": is_flagged,
        "reason": flag_reason,
        "latency": latency_ms
    })
    
    return ChatResponse(
        answer=final_answer,
        confidence_flag=is_critical_flag,
        flag_reason=flag_reason,
        model_used=model_name
    )

if __name__ == "__main__":
    async def run_terminal_test():
        # Clear screen for a clean production look
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*60)
        print("🚀 CLEARPATH AI: BACKEND PRODUCTION READY 🚀")
        print("="*60)
        
        while True:
            user_input = input("\nENTER QUERY (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue
                
            try:
                start_test = time.time()
                # Process through the API endpoint logic
                response = await chat_endpoint(ChatRequest(query=user_input))
                
                print(f"\n[AI RESPONSE] — Model: {response.model_used}")
                print("-" * 55)
                print(response.answer)
                print("-" * 55)
                
                if response.confidence_flag:
                    print(f"🚩 FLAG: {response.flag_reason}")
                else:
                    print(f"✅ STATUS: {response.flag_reason}")
                
                print(f"⏱️ LATENCY: {round((time.time() - start_test)*1000)}ms")

            except Exception as e:
                print(f"❌ ERROR: {str(e)}")

    asyncio.run(run_terminal_test())
import time
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Imports from your other modules ---
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
    model_used: str # Added for visibility

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    query = request.query
    
    # 1. Layer 1: Retrieval (Now standardized to return a list of dicts)
    retrieved_chunks = retrieve_context(query)
    
    # 2. Layer 2: Routing (Uses our new Weighted Scoring system)
    classification, model_name, route_reason = classify_and_route_query(query, retrieved_chunks)
    
    # 3. Layer 3: LLM Generation
    llm_answer, tokens_input, tokens_output = generate_response(query, retrieved_chunks, model_name)
    
    # 4. Layer 4: Evaluation
    is_flagged, flag_reason = evaluate_output(query, llm_answer, retrieved_chunks)
    
    final_answer = llm_answer
    if is_flagged:
        final_answer = f"⚠️ Low confidence — please verify with support.\n\n{llm_answer}"
        
    latency_ms = round((time.time() - start_time) * 1000)
    
    # Log detailed routing info for your debugging
    router_log.info({
        "query": query, 
        "classification": classification, 
        "model": model_name,
        "reason": route_reason,
        "latency_ms": latency_ms
    })
    
    return ChatResponse(
        answer=final_answer,
        confidence_flag=is_flagged,
        flag_reason=flag_reason,
        model_used=model_name
    )

if __name__ == "__main__":
    async def run_terminal_test():
        print("\n" + "="*60)
        print("🚀 CLEARPATH AI: FULL PIPELINE DEBUGGER 🚀")
        print("="*60)
        
        while True:
            user_input = input("\nENTER QUERY (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']: break
                
            try:
                # Run the endpoint logic
                start_test = time.time()
                
                # We fetch chunks here just for the debug print visibility
                chunks = retrieve_context(user_input)
                
                print(f"\n🔍 [STEP 1: RETRIEVED {len(chunks)} CHUNKS]")
                for i, c in enumerate(chunks[:3]):
                    # Match keys to retrieval.py: 'document' and 'text'
                    source = c.get("document", "Unknown")
                    text = c.get("text", "")[:100].replace("\n", " ")
                    print(f"   {i+1}. [{source}] | {text}...")

                # Now run the actual pipeline
                response = await chat_endpoint(ChatRequest(query=user_input))
                
                print(f"\n🧠 [STEP 2: ROUTING & GENERATION]")
                print(f"   Model Selected: {response.model_used}")
                
                print(f"\n🤖 [AI RESPONSE]:")
                print(f"--------------------------------------------------")
                print(response.answer)
                print(f"--------------------------------------------------")
                
                if response.confidence_flag:
                    print(f"🚩 FLAG REASON: {response.flag_reason}")
                
                print(f"⏱️ TOTAL LATENCY: {round((time.time() - start_test)*1000)}ms")

            except Exception as e:
                print(f"❌ CRITICAL ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_terminal_test())
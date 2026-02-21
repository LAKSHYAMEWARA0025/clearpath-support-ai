import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging

# Assuming these are in your config.py
from config import DOCS_DIR, CHROMA_DB_DIR 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_doc_metadata(filename):
    fname_lower = filename.lower()
    if any(kw in fname_lower for kw in ["troubleshooting", "error"]): return "Customer Support Docs", 1.0
    elif any(kw in fname_lower for kw in ["api", "webhook", "architecture", "deployment", "release"]): return "Technical Documentation", 0.9
    elif any(kw in fname_lower for kw in ["guide", "tutorial", "overview", "catalog", "shortcuts"]): return "Product Documentation", 0.8
    elif any(kw in fname_lower for kw in ["faq", "sla", "onboarding"]): return "Customer Support Docs", 0.7
    elif any(kw in fname_lower for kw in ["pricing", "plan", "matrix"]): return "Pricing & Plans", 0.6
    else: return "Internal Policies & Ops", 0.5

def extract_sections(file_path):
    """
    Reads a PDF and groups text under its structural headings based on font size/boldness.
    RULE 1: Each Question (Heading) creates a strictly isolated chunk.
    """
    doc = fitz.open(file_path)
    sections = []
    current_heading = "General"
    current_text = []
    
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") == 0:  
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text: continue
                        
                        size = span.get("size", 10)
                        is_bold = span.get("flags", 0) & 16 
                        
                        # Trigger: Is this a Question/Heading? (Bold or larger font)
                        if size > 12 or (is_bold and len(text) < 100 and not text.endswith('.')):
                            # Save the PREVIOUS section independently. No merging!
                            if current_text:
                                sections.append({
                                    "heading": current_heading, 
                                    "text": " ".join(current_text)
                                })
                                current_text = []
                            # Start the NEW section
                            current_heading = text
                        else:
                            # Append: This is Answer text
                            current_text.append(text)
                            
    # Append the final section
    if current_text:
        sections.append({"heading": current_heading, "text": " ".join(current_text)})
        
    return sections

def chunk_section_safely(text, max_words=600, overlap_words=50):
    """
    RULE 2: Overflow Handler.
    If the isolated Answer is < 600 words, it returns it as a single intact chunk.
    If the Answer is > 600 words, it splits it safely by sentences.
    """
    words = text.split()
    # Fast pass: If it's under the limit, don't split it at all! Keep it 100% intact.
    if len(words) <= max_words:
        return [text]

    # If it exceeds the limit, chunk it by sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence.split())
        
        if current_length + sentence_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Create overlap
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s.split()) <= overlap_words:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s.split())
                else: break
            
            current_chunk = overlap_chunk
            current_length = overlap_len
        
        current_chunk.append(sentence)
        current_length += sentence_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def ingest_documents():
    if not os.path.exists(DOCS_DIR):
        logger.error(f"Documents directory not found at {DOCS_DIR}")
        return

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # UPGRADED: bge-m3 supports up to 8192 tokens, easily handling our 600-word chunks
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
    
    collection = chroma_client.get_or_create_collection(
        name="clearpath_kb",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} 
    )

    all_chunks = []
    
    logger.info("Starting document parsing...")
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".pdf"): continue
            
        file_path = os.path.join(DOCS_DIR, filename)
        domain, authority = get_doc_metadata(filename)
        
        try:
            sections = extract_sections(file_path)
            doc_chunks = []
            
            for section in sections:
                # Our chunker now respects the 600 limit and only splits if necessary
                text_chunks = chunk_section_safely(section["text"])
                for chunk_text in text_chunks:
                    doc_chunks.append({
                        "text": chunk_text,
                        "document_name": filename,
                        "domain": domain,
                        "section": section["heading"],
                        "authority_score": authority
                    })
                    
            all_chunks.extend(doc_chunks)
            logger.info(f"Parsed '{filename}' -> {len(doc_chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")

    if not all_chunks: return

    logger.info(f"Total chunks generated: {len(all_chunks)}. Beginning vector embedding...")
    
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"document_name": c["document_name"], "domain": c["domain"], "section": c["section"], "authority_score": c["authority_score"]} for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        logger.info(f"Embedded batch {i//batch_size + 1}/{(len(documents)//batch_size) + 1}")

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    ingest_documents()
import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings # <-- Added to fix the telemetry bug
from chromadb.utils import embedding_functions
from config import DOCS_DIR, CHROMA_DB_DIR
import logging

# Set up standard logging for the ingest script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Domain and Authority Mapping Rules
def get_doc_metadata(filename):
    """
    Assigns domains and authority scores based on the assignment rules.
    Troubleshooting (1.0) > API docs (0.9) > User guides (0.8) > FAQ (0.7) > Internal (0.5)
    """
    fname_lower = filename.lower()
    
    if any(kw in fname_lower for kw in ["troubleshooting", "error"]):
        return "Customer Support Docs", 1.0
    elif any(kw in fname_lower for kw in ["api", "webhook", "architecture", "deployment", "release"]):
        return "Technical Documentation", 0.9
    elif any(kw in fname_lower for kw in ["guide", "tutorial", "overview", "catalog", "shortcuts"]):
        return "Product Documentation", 0.8
    elif any(kw in fname_lower for kw in ["faq", "sla", "onboarding"]):
        return "Customer Support Docs", 0.7
    elif any(kw in fname_lower for kw in ["pricing", "plan", "matrix"]):
        return "Pricing & Plans", 0.6
    else:
        # Catch-all for Internal Policies and Internal Operations
        return "Internal Policies & Ops", 0.5

# 2. Document-Aware Extraction (PyMuPDF)
def extract_sections(file_path):
    """
    Reads a PDF and groups text under its structural headings based on font size/boldness.
    """
    doc = fitz.open(file_path)
    sections = []
    current_heading = "General"
    current_text = []
    
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") == 0:  # 0 means it's a text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text:
                            continue
                            
                        text = text.strip()
                        if not text: 
                            continue
                        
                        size = span.get("size", 10)
                        # In MuPDF, flags & 2**4 (16) checks if the text is bold
                        is_bold = span.get("flags", 0) & 16 
                        
                        # Heuristic: If font is larger than typical body text (>12) or bold & short
                        if size > 12 or (is_bold and len(text) < 60 and not text.endswith('.')):
                            if current_text:
                                sections.append({
                                    "heading": current_heading, 
                                    "text": " ".join(current_text)
                                })
                                current_text = []
                            current_heading = text
                        else:
                            current_text.append(text)
                            
    # Append the final section
    if current_text:
        sections.append({
            "heading": current_heading, 
            "text": " ".join(current_text)
        })
        
    return sections

# 3. Recursive Sentence-Safe Chunking
def chunk_section_safely(text, max_words=350, overlap_words=50):
    """
    Splits large sections by sentence boundaries, keeping them under token limits 
    while preserving a backward overlap for context.
    """
    # Split by sentence boundaries (. ? !) followed by a space
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence.split())
        
        # If adding this sentence pushes us over the limit, save the chunk
        if current_length + sentence_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Create overlap by taking the last few sentences of the current chunk
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s.split()) <= overlap_words:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s.split())
                else:
                    break
            
            current_chunk = overlap_chunk
            current_length = overlap_len
        
        current_chunk.append(sentence)
        current_length += sentence_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# 4. Main Ingestion Pipeline
def ingest_documents():
    if not os.path.exists(DOCS_DIR):
        logger.error(f"Documents directory not found at {DOCS_DIR}")
        return

    # Initialize ChromaDB locally WITH TELEMETRY DISABLED
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Use the local bge-small-en-v1.5 embedding model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    collection = chroma_client.get_or_create_collection(
        name="clearpath_kb",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} 
    )

    all_chunks = []
    
    logger.info("Starting document parsing...")
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(DOCS_DIR, filename)
        domain, authority = get_doc_metadata(filename)
        
        try:
            sections = extract_sections(file_path)
            doc_chunks = []
            
            for section in sections:
                # Apply our recursive sentence-safe chunker to each section
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
            logger.info(f"Parsed '{filename}' -> Domain: {domain} | {len(doc_chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")

    if not all_chunks:
        logger.warning("No chunks were generated. Are your PDFs empty?")
        return

    logger.info(f"Total chunks generated: {len(all_chunks)}. Beginning vector embedding...")
    
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"document_name": c["document_name"], "domain": c["domain"], "section": c["section"], "authority_score": c["authority_score"]} for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # Batch add to ChromaDB
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        logger.info(f"Embedded batch {i//batch_size + 1}/{(len(documents)//batch_size) + 1}")

    logger.info("Ingestion complete! Database is ready for Layer 1 RAG.")

if __name__ == "__main__":
    ingest_documents()
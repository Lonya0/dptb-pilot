import os
import glob
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
import pypdf

# Configuration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "deeptb_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_files(repo_path: str) -> List[str]:
    """
    Get all relevant files (Markdown, Python, and PDF) from the repository.
    """
    files = []
    # Walk through the directory
    for root, _, filenames in os.walk(repo_path):
        # Skip hidden directories and tests
        if "/." in root or "/tests" in root or "/__pycache__" in root:
            continue
            
        for filename in filenames:
            if filename.endswith(".md") or filename.endswith(".py") or filename.endswith(".pdf"):
                files.append(os.path.join(root, filename))
    return files

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple text chunking with overlap.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_knowledge_base(repo_path: str):
    """
    Build the ChromaDB knowledge base from the repository.
    """
    print(f"Building knowledge base from: {repo_path}")
    print(f"Database path: {CHROMA_DB_DIR}")
    
    # Ensure data directory exists
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Initialize embedding function
    # We use a local sentence-transformer model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Get or create collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection not found or could not be deleted (this is normal for first run): {e}")
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    
    # Get files
    files = get_files(repo_path)
    print(f"Found {len(files)} files to process.")
    
    documents = []
    metadatas = []
    ids = []
    
    count = 0
    for file_path in files:
        try:
            content = ""
            if file_path.endswith(".pdf"):
                try:
                    reader = pypdf.PdfReader(file_path)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
                    continue
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
            # Create relative path for metadata
            rel_path = os.path.relpath(file_path, repo_path)
            if file_path.endswith(".md"):
                file_type = "documentation"
            elif file_path.endswith(".py"):
                file_type = "source_code"
            elif file_path.endswith(".pdf"):
                file_type = "pdf_article"
            else:
                file_type = "unknown"
            
            # Chunk content
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": rel_path,
                    "type": file_type,
                    "chunk_id": i
                })
                ids.append(f"{rel_path}_{i}")
                
                count += 1
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    # Add to collection in batches
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"Adding {len(documents)} chunks to database in {total_batches} batches...")
    
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Processed batch {i//batch_size + 1}/{total_batches}")
        
    print("Knowledge base built successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DeePTB Knowledge Base")
    parser.add_argument("repo_path", help="Path to the DeePTB repository")
    args = parser.parse_args()
    
    build_knowledge_base(args.repo_path)

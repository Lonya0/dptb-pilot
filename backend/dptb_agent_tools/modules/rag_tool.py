import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

# Configuration (must match build_knowledge_base.py)
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
COLLECTION_NAME = "deeptb_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize ChromaDB client (lazy loading to avoid startup overhead if not used)
_client = None
_collection = None
_embedding_function = None

def get_collection():
    global _client, _collection, _embedding_function
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
            _collection = _client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=_embedding_function
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return None
    return _collection

from dptb_agent_tools.init_mcp import mcp

@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Search the DeePTB knowledge base for relevant information.
    
    Args:
        query: The question or topic to search for.
        n_results: Number of results to return (default: 3).
        
    Returns:
        A string containing the relevant context found in the documentation and source code.
    """
    collection = get_collection()
    if not collection:
        return "Error: Knowledge base not initialized. Please run build_knowledge_base.py first."
        
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant information found in the knowledge base."
            
        context = "Found the following relevant information in DeePTB documentation/code:\n\n"
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            source = metadata.get('source', 'unknown')
            file_type = metadata.get('type', 'unknown')
            
            context += f"--- Source: {source} ({file_type}) ---\n"
            context += f"{doc}\n\n"
            
        return context
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# Register the tool with FastMCP if using that directly, 
# but here we are likely using the dynamic loader in init_mcp.py or similar.
# The loader typically looks for functions in the module.
# We expose the function directly.

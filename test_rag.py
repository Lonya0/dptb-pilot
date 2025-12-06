import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dptb_agent_tools.modules.rag_tool import search_knowledge_base

def test_rag():
    query = "How to calculate band structure?"
    print(f"Querying: {query}")
    result = search_knowledge_base(query)
    print("\nResult:")
    print(result)
    
    if "Found the following relevant information" in result:
        print("\n✅ RAG Test Passed!")
    else:
        print("\n❌ RAG Test Failed!")

if __name__ == "__main__":
    test_rag()

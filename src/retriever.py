"""
Document Retriever - Q2: Recherche documentaire dans la base vectorielle
Handles querying the vector database and returning relevant documents with scores.
"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex


class DocumentRetriever:
    """
    Q2: Document retrieval system for querying the vector database.
    Returns relevant documents with affinity scores.
    """
    
    def __init__(self, index: VectorStoreIndex):
        """
        Initialize the document retriever.
        
        Args:
            index: VectorStoreIndex instance from the indexer
        """
        self.index = index
        self.retriever = None
    
    def setup_retriever(self, similarity_top_k: int = 5):
        """
        Setup the retriever with specified parameters.
        
        Args:
            similarity_top_k: Number of most similar documents to retrieve
        """
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        print(f"🔍 Retriever configured to return top {similarity_top_k} documents")
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database and return relevant documents with scores.
        
        Args:
            query: User's search query
            k: Number of documents to retrieve
            
        Returns:
            List of documents with metadata and affinity scores
        """
        if not self.retriever or self.retriever.similarity_top_k != k:
            self.setup_retriever(similarity_top_k=k)
        
        print(f"🔍 Searching for: '{query}'")
        
        # Retrieve nodes from vector database
        nodes = self.retriever.retrieve(query)
        
        # Format results with scores and metadata
        results = []
        for i, node in enumerate(nodes, 1):
            # Extract source information
            source_path = node.metadata.get('file_name', 'Unknown')
            if 'data' in source_path:
                source_path = source_path.replace('data\\', '').replace('data/', '')
            
            # Calculate similarity scores
            # Note: LlamaIndex returns distance scores (lower = more similar)
            # We convert to similarity scores (higher = more similar)
            distance_score = getattr(node, 'score', 0.0)
            similarity_score = max(0.0, min(1.0, 1.0 - distance_score))
            
            result = {
                "rank": i,
                "content": node.text,
                "source": source_path,
                "page": node.metadata.get('page_label', 'N/A'),
                "distance_score": round(distance_score, 4),
                "similarity_score": round(similarity_score, 4),
                "similarity_percent": round(similarity_score * 100, 1),
                "metadata": node.metadata
            }
            
            results.append(result)
        
        print(f"✅ Found {len(results)} relevant documents")
        return results
    
    def get_top_documents(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Get top relevant documents with summary statistics.
        
        Args:
            query: Search query
            k: Number of top documents
            
        Returns:
            Dictionary with documents and summary stats
        """
        documents = self.search_documents(query, k)
        
        if not documents:
            return {
                "query": query,
                "documents": [],
                "count": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0
            }
        
        # Calculate summary statistics
        similarities = [doc["similarity_score"] for doc in documents]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        
        return {
            "query": query,
            "documents": documents,
            "count": len(documents),
            "avg_similarity": round(avg_similarity, 3),
            "max_similarity": round(max_similarity, 3)
        }
    
    def test_queries(self, test_queries: List[str], k: int = 3) -> Dict[str, Any]:
        """
        Test multiple queries and return results.
        
        Args:
            test_queries: List of queries to test
            k: Number of documents per query
            
        Returns:
            Dictionary with all test results
        """
        print(f"\n🧪 Testing {len(test_queries)} queries...")
        
        results = {}
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}/{len(test_queries)} ---")
            result = self.get_top_documents(query, k)
            results[query] = result
            
            print(f"Query: {query}")
            print(f"Found: {result['count']} documents")
            print(f"Avg similarity: {result['avg_similarity']:.1%}")
            print(f"Max similarity: {result['max_similarity']:.1%}")
        
        return results
    
    def print_search_results(self, query: str, k: int = 5, show_content: bool = True):
        """
        Print formatted search results for debugging.
        
        Args:
            query: Search query
            k: Number of results
            show_content: Whether to show document content
        """
        results = self.search_documents(query, k)
        
        print(f"\n" + "="*80)
        print(f"SEARCH RESULTS FOR: '{query}'")
        print("="*80)
        
        if not results:
            print("No documents found.")
            return
        
        for doc in results:
            print(f"\n📄 Rank {doc['rank']} - {doc['source']} (Page {doc['page']})")
            print(f"   Similarity: {doc['similarity_percent']}% (Score: {doc['similarity_score']:.3f})")
            
            if show_content:
                content = doc['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"   Content: {content}")
        
        print(f"\n" + "="*80)

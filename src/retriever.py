"""
Q2: Document Retrieval System
==============================
This module handles document retrieval from the vector database.

Features:
- Query the vector database with user queries
- Return the most relevant documents with similarity scores
- Support for configurable number of results (top-k)
"""

from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class DocumentRetriever:
    """
    Q2: Document retrieval system for querying the vector database.
    
    This class allows searching for relevant documents based on user queries,
    returning both document content and similarity scores (affinity scores).
    """
    
    def __init__(self,
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 collection_name: str = "rag_collection"):
        """
        Initialize the document retriever.
        
        Args:
            vectorstore_dir: Directory containing the ChromaDB vector store
            embedding_model_name: HuggingFace embedding model (must match indexer)
            collection_name: Name of the ChromaDB collection
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize embedding model (must match the one used for indexing)
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        print(f"✅ Embedding model loaded")
        
        # Initialize ChromaDB client
        if not self.vectorstore_dir.exists():
            raise ValueError(f"Vector store not found: {vectorstore_dir}. Please build index first.")
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        
        # Load the index
        self.index = self._load_index()
    
    def _load_index(self) -> VectorStoreIndex:
        """
        Load the vector index from ChromaDB.
        
        Returns:
            VectorStoreIndex: The loaded index
        """
        print(f"📂 Loading index from {self.vectorstore_dir}...")
        
        # Get the ChromaDB collection
        chroma_collection = self.chroma_client.get_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        print(f"✅ Index loaded successfully")
        return index
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Q2: Search for relevant documents in the vector database.
        
        This function queries the vector database and returns:
        - A list of the most relevant documents
        - Affinity scores (similarity scores) for each document
        
        Args:
            query: User's search query
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing:
                - rank: Position in results (1-indexed)
                - content: Document text content
                - source: Source file name
                - page: Page number (if available)
                - score: Raw distance score (lower = better)
                - similarity: Similarity score (0-1, higher = better)
                - similarity_percent: Similarity as percentage
                - metadata: Additional metadata
        """
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Retrieving top {k} results...\n")
        
        # Create retriever from index
        retriever = self.index.as_retriever(similarity_top_k=k)
        
        # Retrieve relevant nodes (chunks)
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for i, node in enumerate(nodes, 1):
            # Extract metadata
            source_path = node.metadata.get('file_name', 'Unknown')
            # Clean up path (remove data/ prefix)
            if 'data' in source_path:
                source_path = source_path.replace('data\\', '').replace('data/', '')
            
            # Calculate similarity score
            # LlamaIndex returns distance scores (lower = more similar)
            # We convert to similarity (0-1, higher = more similar)
            raw_score = getattr(node, 'score', 0.0)
            similarity = max(0.0, min(1.0, 1.0 - raw_score))
            
            # Extract advanced metadata if available
            title = node.metadata.get('title', None)
            questions_answered = node.metadata.get('questions_this_excerpt_can_answer', None)
            
            result = {
                "rank": i,
                "content": node.text,
                "source": source_path,
                "page": node.metadata.get('page_label', 'N/A'),
                "score": round(raw_score, 4),  # Raw distance score
                "similarity": round(similarity, 4),  # Converted similarity
                "similarity_percent": round(similarity * 100, 1),
                "metadata": node.metadata
            }
            
            # Add advanced RAG metadata if available
            if title:
                result["title"] = title
            if questions_answered:
                result["questions_answered"] = questions_answered
            
            results.append(result)
        
        return results
    
    def search_and_print_results(self, query: str, k: int = 5):
        """
        Q2: Search and print results in a formatted way.
        
        This is a convenience method that searches and displays results
        in a user-friendly format.
        
        Args:
            query: User's search query
            k: Number of results to return
        """
        results = self.search(query, k=k)
        
        if not results:
            print("❌ No results found.")
            return
        
        print("="*80)
        print("📊 SEARCH RESULTS")
        print("="*80 + "\n")
        
        for result in results:
            print(f"Rank #{result['rank']}")
            print(f"Source: {result['source']} (Page: {result['page']})")
            print(f"Similarity: {result['similarity_percent']}% (score: {result['score']})")
            
            # Show advanced metadata if available
            if result.get('title'):
                print(f"Title: {result['title']}")
            if result.get('questions_answered'):
                qa = result['questions_answered']
                if isinstance(qa, str):
                    print(f"Questions Answered: {qa}")
                elif isinstance(qa, list):
                    print(f"Questions Answered: {', '.join(qa)}")
            
            print(f"\nContent Preview:")
            # Show first 300 characters
            content = result['content']
            if len(content) > 300:
                print(f"{content[:300]}...")
            else:
                print(content)
            print("\n" + "-"*80 + "\n")
        
        # Summary statistics
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        print(f"📈 Average Similarity: {avg_similarity*100:.1f}%")
        print(f"📚 Total Results: {len(results)}")


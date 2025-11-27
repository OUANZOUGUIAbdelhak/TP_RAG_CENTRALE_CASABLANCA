"""
Document Embedder - Embedding component for RAG system
Handles text embedding using HuggingFace models.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode


class DocumentEmbedder:
    """
    Document embedder using HuggingFace models.
    Part of Q1 indexation pipeline - Embedding step.
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str = "cpu",
                 max_length: int = 512):
        """
        Initialize document embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run model on ('cpu' or 'cuda')
            max_length: Maximum sequence length for embeddings
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        print(f"🔧 Loading embedding model: {model_name}")
        
        # Initialize HuggingFace embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            max_length=max_length
        )
        
        print(f"✅ Embedding model loaded successfully")
        
        # Get embedding dimension
        self.embedding_dim = self._get_embedding_dimension()
        print(f"📏 Embedding dimension: {self.embedding_dim}")
    
    def _get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension
        """
        try:
            # Test with a simple text to get dimension
            test_embedding = self.embed_model.get_text_embedding("test")
            return len(test_embedding)
        except Exception as e:
            print(f"⚠️  Could not determine embedding dimension: {e}")
            return 384  # Default for many models
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"🔢 Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"   Progress: {i}/{len(texts)} embeddings generated")
            
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(text)
            embeddings.append(embedding)
        
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        Generate embeddings for text nodes and attach them.
        
        Args:
            nodes: List of text nodes to embed
            
        Returns:
            List of nodes with embeddings attached
        """
        print(f"🔢 Generating embeddings for {len(nodes)} nodes...")
        
        # Extract texts from nodes
        texts = [node.text for node in nodes]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Attach embeddings to nodes
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            
            # Add embedding metadata
            node.metadata['embedding_model'] = self.model_name
            node.metadata['embedding_dim'] = len(embedding)
        
        return nodes
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding vector
        """
        return self.embed_model.get_query_embedding(query)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, 
                         query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of similarity results with scores and indices
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'embedding_dimension': self.embedding_dim,
            'model_type': 'HuggingFace'
        }
    
    def benchmark_embedding_speed(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark embedding generation speed.
        
        Args:
            sample_texts: Sample texts for benchmarking
            
        Returns:
            Benchmark results
        """
        import time
        
        print(f"⏱️  Benchmarking embedding speed with {len(sample_texts)} samples...")
        
        start_time = time.time()
        embeddings = self.embed_texts(sample_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        texts_per_second = len(sample_texts) / total_time
        
        return {
            'total_texts': len(sample_texts),
            'total_time_seconds': round(total_time, 2),
            'texts_per_second': round(texts_per_second, 2),
            'avg_time_per_text': round(total_time / len(sample_texts), 4),
            'embedding_dimension': len(embeddings[0]) if embeddings else 0
        }

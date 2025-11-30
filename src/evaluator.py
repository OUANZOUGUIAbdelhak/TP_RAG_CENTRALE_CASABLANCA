"""
Q4: RAG/LLM System Evaluation
==============================
This module implements evaluation mechanisms for the RAG system.

Features: 
- Evaluate retrieval quality (relevance of retrieved documents)
- Evaluate answer quality (LLM response quality)
- Compute various metrics (precision, relevance, etc.)
- End-to-end evaluation with test cases
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Handle both package and direct imports
try:
    from .qa_system import QASystem
except ImportError:
    from qa_system import QASystem


class RAGEvaluator:
    """
    Q4: Evaluation system for RAG/LLM performance.
    
    This class implements mechanisms to evaluate:
    - Retrieval quality: Are the right documents being retrieved?
    - Answer quality: Are the LLM answers relevant and accurate?
    - Overall system performance
    """
    
    def __init__(self,
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                 collection_name: str = "rag_collection"):
        """
        Initialize the evaluator.
        
        Args:
            vectorstore_dir: Directory containing the ChromaDB vector store
            embedding_model_name: HuggingFace embedding model name
            collection_name: ChromaDB collection name
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize QA system for evaluation
        self.qa_system = QASystem(
            vectorstore_dir=str(vectorstore_dir),
            embedding_model_name=embedding_model_name,
            collection_name=collection_name
        )
        
        # Load index for retrieval evaluation
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        self.index = self._load_index()
    
    def _load_index(self) -> VectorStoreIndex:
        """Load the vector index from ChromaDB."""
        if not self.vectorstore_dir.exists():
            raise ValueError(f"Vector store not found: {self.vectorstore_dir}")
        
        chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        chroma_collection = chroma_client.get_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
    
    def evaluate_retrieval(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Q4: Evaluate retrieval quality for a given query.
        
        Metrics:
        - Average similarity score of retrieved documents
        - Score distribution
        - Number of highly relevant documents (similarity > 0.7)
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieval metrics
        """
        # Retrieve documents
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        
        # Calculate metrics
        similarities = []
        for node in nodes:
            score = getattr(node, 'score', 0.0)
            similarity = max(0.0, min(1.0, 1.0 - score))
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        highly_relevant = sum(1 for s in similarities if s > 0.7)
        
        return {
            "query": query,
            "num_retrieved": len(nodes),
            "average_similarity": round(avg_similarity, 3),
            "highly_relevant_count": highly_relevant,
            "similarities": [round(s, 3) for s in similarities],
            "min_similarity": round(min(similarities), 3) if similarities else 0.0,
            "max_similarity": round(max(similarities), 3) if similarities else 0.0
        }
    
    def evaluate_answer_quality(self, 
                                 question: str, 
                                 expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Q4: Evaluate answer quality for a given question.
        
        Metrics:
        - Answer length (longer answers often indicate more detail)
        - Keyword presence (if expected keywords provided)
        - Confidence score (from retrieval)
        - Source diversity (number of different sources used)
        
        Args:
            question: Question to answer
            expected_keywords: Optional list of keywords expected in answer
            
        Returns:
            Dictionary with answer quality metrics
        """
        # Get answer from QA system
        result = self.qa_system.answer(question)
        
        # Calculate metrics
        answer = result["answer"]
        sources = result["sources"]
        
        # Basic metrics
        answer_length = len(answer)
        num_sources = len(sources)
        unique_sources = len(set(s["source"] for s in sources))
        confidence = result["confidence"]
        
        # Keyword presence (if provided)
        keyword_score = 0.0
        if expected_keywords:
            keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
            keyword_score = keywords_found / len(expected_keywords)
        
        return {
            "question": question,
            "answer_length": answer_length,
            "num_sources": num_sources,
            "unique_sources": unique_sources,
            "confidence": confidence,
            "keyword_score": round(keyword_score, 3) if expected_keywords else None,
            "keywords_expected": expected_keywords,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
        }
    
    def evaluate_end_to_end(self, test_cases: List[Dict[str, Any]]):
        """
        Q4: Comprehensive end-to-end evaluation.
        
        Run multiple test cases and aggregate metrics.
        
        Args:
            test_cases: List of test case dictionaries with:
                - question: The question to ask
                - expected_keywords: Optional list of expected keywords
        """
        print("\n" + "="*80)
        print("📊 Q4: COMPREHENSIVE SYSTEM EVALUATION")
        print("="*80 + "\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case['question']
            keywords = test_case.get('expected_keywords', [])
            
            print(f"Test Case #{i}: {question}")
            print("-"*80)
            
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(question, k=10)
            print(f"📥 Retrieval:")
            print(f"   - Average Similarity: {retrieval_metrics['average_similarity']*100:.1f}%")
            print(f"   - Highly Relevant: {retrieval_metrics['highly_relevant_count']}/10")
            
            # Evaluate answer
            answer_metrics = self.evaluate_answer_quality(question, keywords)
            print(f"\n💡 Answer Quality:")
            print(f"   - Answer Length: {answer_metrics['answer_length']} characters")
            print(f"   - Confidence: {answer_metrics['confidence']*100:.1f}%")
            print(f"   - Sources Used: {answer_metrics['num_sources']} ({answer_metrics['unique_sources']} unique)")
            if answer_metrics['keyword_score'] is not None:
                print(f"   - Keyword Score: {answer_metrics['keyword_score']*100:.1f}%")
            
            print("\n")
            
            results.append({
                "test_case": i,
                "question": question,
                "retrieval": retrieval_metrics,
                "answer": answer_metrics
            })
        
        # Aggregate statistics
        print("="*80)
        print("📈 AGGREGATE STATISTICS")
        print("="*80 + "\n")
        
        avg_retrieval_sim = sum(r['retrieval']['average_similarity'] for r in results) / len(results)
        avg_confidence = sum(r['answer']['confidence'] for r in results) / len(results)
        avg_answer_length = sum(r['answer']['answer_length'] for r in results) / len(results)
        
        print(f"Average Retrieval Similarity: {avg_retrieval_sim*100:.1f}%")
        print(f"Average Answer Confidence: {avg_confidence*100:.1f}%")
        print(f"Average Answer Length: {avg_answer_length:.0f} characters")
        
        # Keyword scores (if available)
        keyword_scores = [r['answer']['keyword_score'] for r in results if r['answer']['keyword_score'] is not None]
        if keyword_scores:
            avg_keyword = sum(keyword_scores) / len(keyword_scores)
            print(f"Average Keyword Score: {avg_keyword*100:.1f}%")
    
    def quick_quality_check(self):
        """
        Q4: Quick quality check with sample queries.
        
        Run a few sample queries to quickly assess system quality.
        """
        print("\n" + "="*80)
        print("⚡ Q4: QUICK QUALITY CHECK")
        print("="*80 + "\n")
        
        sample_queries = [
            "What is the main topic of these documents?",
            "Can you summarize the key points?",
            "which document talks more about thiamine deficiency in developed countires?"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"Query #{i}: {query}")
            print("-"*80)
            
            metrics = self.evaluate_retrieval(query, k=3)
            print(f"Retrieval Quality: {metrics['average_similarity']*100:.1f}%")
            print(f"Highly Relevant: {metrics['highly_relevant_count']}/3")
            print()
        
        print("✅ Quick check complete!")


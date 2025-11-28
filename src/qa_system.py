"""
Q3: Question-Answering System with LLM
=======================================
This module implements the Q&A system using an LLM.

Features:
- Uses retrieved documents from vector database as context
- Employs an open-source LLM (via Groq API)
- Custom prompt template for optimal answer generation
- Synthesizes information from multiple sources
"""

import os
# Disable PostHog telemetry (prevents timeout errors)
os.environ["LLAMA_TELEMETRY_DISABLED"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from pathlib import Path
from typing import Dict, Any, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import yaml


class QASystem:
    """
    Q3: Question-Answering system using LLM with RAG.
    
    This class combines document retrieval with LLM-based answer generation.
    It uses:
    - Retrieved paragraphs from the vector database as context
    - An open-source LLM (Groq) to synthesize answers
    - A custom prompt template optimized for RAG
    """
    
    def __init__(self,
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 collection_name: str = "rag_collection",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 use_gemini: bool = False):
        """
        Initialize the Q&A system.
        
        Args:
            vectorstore_dir: Directory containing the ChromaDB vector store
            embedding_model_name: HuggingFace embedding model name
            collection_name: ChromaDB collection name
            groq_api_key: Groq API key for LLM (if None, loads from config)
            groq_model: Groq model name to use
            use_gemini: Whether to use Gemini instead of Groq (not implemented)
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Load config to get API key if not provided
        if groq_api_key is None:
            groq_api_key = self._load_groq_api_key()
        
        # Initialize embedding model
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        print(f"✅ Embedding model loaded")
        
        # Initialize LLM (Groq - open-source models)
        self.llm = None
        if groq_api_key:
            try:
                print(f"🔧 Initializing Groq LLM: {groq_model}")
                self.llm = Groq(model=groq_model, api_key=groq_api_key)
                print(f"✅ Groq LLM initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize Groq LLM: {e}")
                self.llm = None
        else:
            print("⚠️  No Groq API key found. LLM queries will not work.")
        
        # Load index
        self.index = self._load_index()
        
        # Create query engine with custom prompt
        self.query_engine = None
        if self.llm:
            self.query_engine = self._create_query_engine()
    
    def _load_groq_api_key(self) -> Optional[str]:
        """Load Groq API key from Config.yaml."""
        try:
            # Try multiple possible paths for Config.yaml
            config_paths = [
                Path(__file__).parent.parent / "Config.yaml",  # src/../Config.yaml
                Path(__file__).parent.parent.parent / "Config.yaml",  # src/../../Config.yaml
                Path("Config.yaml"),  # Current directory
                Path(__file__).parent / "Config.yaml",  # src/Config.yaml (unlikely)
            ]
            
            for config_path in config_paths:
                abs_path = config_path.resolve()
                if abs_path.exists():
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    api_key = config.get('groq', {}).get('api_key')
                    if api_key:
                        return api_key
        except Exception as e:
            print(f"⚠️  Could not load Groq API key from config: {e}")
        return None
    
    def _load_index(self) -> VectorStoreIndex:
        """Load the vector index from ChromaDB."""
        if not self.vectorstore_dir.exists():
            raise ValueError(f"Vector store not found: {self.vectorstore_dir}. Please build index first.")
        
        print(f"📂 Loading index from {self.vectorstore_dir}...")
        
        chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        chroma_collection = chroma_client.get_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        print(f"✅ Index loaded successfully")
        return index
    
    def _create_query_engine(self):
        """
        Q3: Create query engine with custom prompt template.
        
        The prompt template is optimized for RAG and includes:
        - Context from retrieved documents
        - Instructions for synthesizing information
        - Guidelines for handling insufficient context
        """
        # Custom prompt template for Q3
        # This template ensures the LLM:
        # 1. Uses only the provided context (retrieved documents)
        # 2. Uses metadata (titles, questions) internally for better understanding
        # 3. Provides well-formatted, ChatGPT-style responses
        qa_prompt_template = PromptTemplate(
            "You are a helpful assistant that provides clear, well-structured answers based on the provided context.\n\n"
            "Context information from the knowledge base:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"
            "IMPORTANT INSTRUCTIONS:\n\n"
            "📋 CONTENT GUIDELINES:\n"
            "• Use ONLY the context information above to answer the question\n"
            "• Use metadata (titles, questions answered) internally to better understand the context, but don't explicitly mention them as metadata fields\n"
            "• Synthesize information from multiple sources if needed\n"
            "• If the context does not contain enough information, say so explicitly\n"
            "• Do not use prior knowledge outside the provided context\n\n"
            "✨ FORMATTING REQUIREMENTS:\n"
            "• Use clear headings with ## or ### for main sections\n"
            "• Use bullet points (• or -) for lists and key points\n"
            "• Organize content into logical sections\n"
            "• Add relevant emojis to improve readability (✨ 📝 📦 💡 🔍 ✅ ⚠️ 🎯 📊 🔗 etc.)\n"
            "• Use proper spacing between sections\n"
            "• Make the response engaging, user-friendly, and visually appealing\n"
            "• Structure longer answers with:\n"
            "  - An introduction/overview\n"
            "  - Main points in organized sections\n"
            "  - A brief summary if appropriate\n\n"
            "🚫 DO NOT:\n"
            "• Include metadata fields like 'title' or 'questions_answered' as separate items\n"
            "• Show raw metadata in your response\n"
            "• Use overly technical language unless necessary\n\n"
            "Question: {query_str}\n\n"
            "Answer (formatted nicely with headings, bullet points, emojis, and clear structure): "
        )
        
        try:
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=5,  # Retrieve top 5 most relevant chunks
                response_mode="compact",  # Compact response mode for efficiency
                text_qa_template=qa_prompt_template  # Our custom prompt
            )
            print(f"✅ Query engine created with custom prompt template")
            return query_engine
        except TypeError:
            # Fallback if text_qa_template parameter doesn't work
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=5,
                response_mode="compact"
            )
            print(f"✅ Query engine created (using default prompt)")
            return query_engine
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Q3: Answer a question using RAG (Retrieval + LLM generation).
        
        Process:
        1. Retrieve relevant paragraphs from vector database
        2. Provide context to LLM via custom prompt
        3. LLM synthesizes information and generates answer
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing:
                - answer: Generated answer from LLM
                - sources: List of source documents with scores
                - confidence: Overall confidence score
        """
        if not self.query_engine:
            return {
                "answer": "LLM not available. Please configure Groq API key in Config.yaml",
                "sources": [],
                "confidence": 0.0
            }
        
        print(f"🤔 Processing question: {question}")
        
        try:
            # Query the engine (retrieves context + generates answer)
            response = self.query_engine.query(question)
            
            # Extract answer (LLM-generated response)
            if hasattr(response, 'response'):
                answer = str(response.response).strip()
            else:
                answer = str(response).strip()
            
            # Extract source documents (metadata used internally, not exposed in response)
            source_nodes = getattr(response, 'source_nodes', [])
            sources = []
            for node in source_nodes[:5]:
                source_path = node.metadata.get('file_name', 'Unknown')
                if 'data' in source_path:
                    source_path = source_path.replace('data\\', '').replace('data/', '')
                
                score = getattr(node, 'score', 0.0)
                similarity = max(0.0, min(1.0, 1.0 - score))
                
                # Note: Metadata (title, questions_answered) is available in node.metadata
                # and used by the LLM for context, but we don't expose it in the response
                # The LLM can incorporate this information naturally into the answer content
                
                source = {
                    "source": source_path,
                    "page": node.metadata.get('page_label', 'N/A'),
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "score": round(score, 4),
                    "similarity": round(similarity, 4),
                    "similarity_percent": round(similarity * 100, 1)
                }
                
                # Metadata is NOT added to the response - it's used internally only
                sources.append(source)
            
            # Calculate confidence from source similarities
            confidence = sum(s["similarity"] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def answer_with_details(self, question: str):
        """
        Q3: Answer a question and print detailed results.
        
        Args:
            question: User's question
        """
        result = self.answer(question)
        
        print("\n" + "="*80)
        print("💡 ANSWER")
        print("="*80 + "\n")
        print(result["answer"])
        
        if result["sources"]:
            print("\n" + "="*80)
            print("📚 SOURCES")
            print("="*80 + "\n")
            
            for i, source in enumerate(result["sources"], 1):
                print(f"Source #{i}: {source['source']} (Page: {source['page']})")
                print(f"Similarity: {source['similarity_percent']}%")
                print(f"Content: {source['content']}")
                print("-"*80 + "\n")
                # Note: Metadata (title, questions_answered) is used internally by the LLM
                # but not shown in the response to keep it clean and user-friendly
            
            print(f"🎯 Overall Confidence: {result['confidence']*100:.1f}%")


"""
LlamaIndex-based RAG System
Modern, efficient RAG implementation using LlamaIndex
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class RAGSystem:
    """
    Complete RAG system using LlamaIndex with ChromaDB vector store.
    Handles document loading, indexing, and querying.
    """
    
    def __init__(self, 
                 data_dir: str = "./data",
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the RAG system.
        
        Args:
            data_dir: Directory containing documents
            vectorstore_dir: Directory for ChromaDB persistence
            embedding_model: HuggingFace embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            groq_api_key: Groq API key for LLM
            groq_model: Groq model name
        """
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        print(f"✅ Embedding model loaded")
        
        # Initialize ChromaDB
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        self.collection_name = "rag_collection"
        
        # Initialize LLM (Groq)
        self.llm = None
        if groq_api_key:
            try:
                print(f"🔧 Initializing Groq LLM: {groq_model}")
                self.llm = Groq(model=groq_model, api_key=groq_api_key)
                print(f"✅ Groq LLM initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize Groq LLM: {e}")
                import traceback
                print(f"⚠️  Error details: {traceback.format_exc()}")
        else:
            print("⚠️  No Groq API key provided. LLM queries will not work.")
            print("⚠️  Without an LLM, queries will return raw document content instead of generated answers.")
        
        # Initialize index (will be built or loaded)
        self.index = None
        self.query_engine = None
        
    def build_index(self):
        """
        Build the vector index from documents in data_dir.
        """
        print("\n" + "="*80)
        print("🚀 BUILDING INDEX WITH LLAMAINDEX")
        print("="*80 + "\n")
        
        # Step 1: Load documents
        print("📂 Step 1: Loading documents...")
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            filename_as_id=True
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"   ✅ Loaded {len(documents)} document(s)")
        
        # Step 2: Configure document metadata
        print("\n📝 Step 2: Configuring document metadata...")
        for doc in documents:
            # Set text template for better embedding
            doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
            # Exclude page_label from embeddings (not useful for semantic search)
            if "page_label" not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append("page_label")
        
        print(f"   ✅ Configured {len(documents)} document(s)")
        
        # Step 3: Create text splitter
        print(f"\n✂️  Step 3: Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = SentenceSplitter(
            separator=" ",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Step 4: Create or get ChromaDB collection
        print(f"\n💾 Step 4: Setting up ChromaDB vector store...")
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"   ✅ ChromaDB collection '{self.collection_name}' ready")
        
        # Step 5: Create index with transformations
        print(f"\n🔨 Step 5: Creating vector index...")
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[text_splitter],
            show_progress=True
        )
        
        print(f"   ✅ Index created successfully!")
        
        # Step 6: Create query engine
        if self.llm:
            print(f"\n🤖 Step 6: Creating query engine with LLM...")
            # Create a proper prompt template that instructs the LLM to generate an answer
            # This ensures the LLM synthesizes an answer rather than just returning context
            qa_prompt_template = PromptTemplate(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above and not prior knowledge, "
                "answer the following question in a clear, structured, and comprehensive manner.\n"
                "Provide a well-formatted answer based solely on the provided context.\n"
                "If the context does not contain enough information to answer the question, "
                "say so explicitly.\n\n"
                "Question: {query_str}\n"
                "Answer: "
            )
            try:
                self.query_engine = self.index.as_query_engine(
                    llm=self.llm,
                    similarity_top_k=5,
                    response_mode="compact",
                    text_qa_template=qa_prompt_template
                )
                print(f"   ✅ Query engine ready with custom prompt template")
            except TypeError as e:
                # Fallback if text_qa_template parameter doesn't work
                print(f"   ⚠️  Trying alternative prompt configuration: {e}")
                try:
                    # Try without explicit template (should use default)
                    self.query_engine = self.index.as_query_engine(
                        llm=self.llm,
                        similarity_top_k=5,
                        response_mode="compact"
                    )
                    print(f"   ✅ Query engine ready (using default prompt)")
                except Exception as e2:
                    print(f"   ❌ Failed to create query engine: {e2}")
                    self.query_engine = None
        else:
            print(f"\n⚠️  Step 6: Skipping query engine (no LLM available)")
        
        print("\n" + "="*80)
        print("✅ INDEX BUILD COMPLETE!")
        print("="*80 + "\n")
        
    def load_index(self):
        """
        Load existing index from vectorstore.
        """
        try:
            print(f"📂 Loading existing index from {self.vectorstore_dir}...")
            
            # Get ChromaDB collection
            chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            
            # Create query engine if LLM is available
            if self.llm:
                # Create a proper prompt template that instructs the LLM to generate an answer
                qa_prompt_template = PromptTemplate(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above and not prior knowledge, "
                    "answer the following question in a clear, structured, and comprehensive manner.\n"
                    "Provide a well-formatted answer based solely on the provided context.\n"
                    "If the context does not contain enough information to answer the question, "
                    "say so explicitly.\n\n"
                    "Question: {query_str}\n"
                    "Answer: "
                )
                try:
                    self.query_engine = self.index.as_query_engine(
                        llm=self.llm,
                        similarity_top_k=5,
                        response_mode="compact",
                        text_qa_template=qa_prompt_template
                    )
                except TypeError:
                    # Fallback if text_qa_template parameter doesn't work
                    self.query_engine = self.index.as_query_engine(
                        llm=self.llm,
                        similarity_top_k=5,
                        response_mode="compact"
                    )
            
            print(f"✅ Index loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            return False
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.index:
            if not self.load_index():
                return {
                    "answer": "Index not built. Please build the index first.",
                    "sources": [],
                    "confidence": 0.0
                }
        
        if not self.query_engine:
            # Fallback: retrieve documents without LLM
            retriever = self.index.as_retriever(similarity_top_k=k)
            nodes = retriever.retrieve(question)
            
            sources = []
            for i, node in enumerate(nodes):
                source_path = node.metadata.get('file_name', 'Unknown')
                sources.append({
                    "source": source_path,
                    "page": node.metadata.get('page_label', 'N/A'),
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "score": getattr(node, 'score', 0.0),
                    "similarity": max(0.0, min(1.0, 1.0 - getattr(node, 'score', 1.0))),
                    "similarity_percent": round(max(0.0, min(1.0, 1.0 - getattr(node, 'score', 1.0))) * 100, 1)
                })
            
            # Simple answer from top result
            answer = nodes[0].text if nodes else "No relevant documents found."
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": sources[0]["similarity"] if sources else 0.0
            }
        
        # Use LLM query engine
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized. LLM may not be configured.")
            
            print(f"🔍 Querying with LLM: {question[:100]}...")
            response = self.query_engine.query(question)
            
            # Extract the answer from the response
            # LlamaIndex Response objects: response.response contains the actual answer text
            # Converting to string should give us the answer, but let's be explicit
            if hasattr(response, 'response'):
                # This is the proper way - response.response contains the LLM-generated answer
                answer = str(response.response).strip()
            elif hasattr(response, '__str__'):
                # Fallback: convert response to string
                answer = str(response).strip()
            else:
                answer = str(response).strip()
            
            # Debug: Check if we got a proper answer
            print(f"📝 LLM Response length: {len(answer)} characters")
            if len(answer) < 50:
                print(f"⚠️  Warning: Response seems too short: {answer[:200]}")
            
            # Verify this is an actual answer, not raw document content
            # Raw content typically doesn't address the question directly
            if len(answer) > 1000 and question.lower() not in answer.lower()[:200]:
                print(f"⚠️  Warning: Response might be raw content. Checking...")
                # Try to extract from response.response if available
                if hasattr(response, 'response'):
                    answer = str(response.response).strip()
            
            # Get source nodes
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                source_nodes = response.source_nodes
            elif hasattr(response, 'metadata') and 'source_nodes' in response.metadata:
                source_nodes = response.metadata['source_nodes']
            elif hasattr(response, 'get_formatted_sources'):
                # Try to extract sources from formatted response
                try:
                    formatted = response.get_formatted_sources(length=100)
                    # Parse sources if possible
                except:
                    pass
            
            sources = []
            for node in source_nodes[:k]:
                source_path = node.metadata.get('file_name', 'Unknown')
                # Clean up path
                if 'data' in source_path:
                    source_path = source_path.replace('data\\', '').replace('data/', '')
                
                score = getattr(node, 'score', 0.0)
                similarity = max(0.0, min(1.0, 1.0 - score))
                
                sources.append({
                    "source": source_path,
                    "page": node.metadata.get('page_label', 'N/A'),
                    "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "score": round(score, 4),
                    "similarity": round(similarity, 4),
                    "similarity_percent": round(similarity * 100, 1)
                })
            
            # Calculate confidence from source similarities
            confidence = sum(s["similarity"] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Query error: {e}")
            print(f"❌ Error details: {error_details}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents without LLM (retrieval only).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of document dictionaries with scores
        """
        if not self.index:
            if not self.load_index():
                return []
        
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        
        results = []
        for i, node in enumerate(nodes, 1):
            source_path = node.metadata.get('file_name', 'Unknown')
            if 'data' in source_path:
                source_path = source_path.replace('data\\', '').replace('data/', '')
            
            score = getattr(node, 'score', 0.0)
            similarity = max(0.0, min(1.0, 1.0 - score))
            
            results.append({
                "rank": i,
                "content": node.text,
                "source": source_path,
                "page": node.metadata.get('page_label', 'N/A'),
                "score": round(score, 4),
                "similarity": round(similarity, 4),
                "similarity_percent": round(similarity * 100, 1),
                "metadata": node.metadata
            })
        
        return results


class RAGChatbot:
    """
    Chatbot with conversation memory using LlamaIndex RAG system.
    """
    
    def __init__(self, rag_system: RAGSystem, max_history: int = 5):
        """
        Initialize chatbot.
        
        Args:
            rag_system: RAGSystem instance
            max_history: Maximum conversation history to keep
        """
        self.rag_system = rag_system
        self.max_history = max_history
        self.conversation_history = []
    
    def chat(self, message: str, verbose: bool = False) -> str:
        """
        Process a chat message with conversation context.
        
        Args:
            message: User's message
            verbose: Print debug information
            
        Returns:
            Assistant's response
        """
        # Build contextualized query
        if self.conversation_history:
            # Add recent context
            recent_context = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in self.conversation_history[-self.max_history:]
            ])
            contextualized_query = f"Previous conversation:\n{recent_context}\n\nCurrent question: {message}"
        else:
            contextualized_query = message
        
        if verbose:
            print(f"💬 Query: {message}")
            if contextualized_query != message:
                print(f"📝 Contextualized: {contextualized_query[:200]}...")
        
        # Query RAG system
        result = self.rag_system.query(contextualized_query)
        
        # Store in history
        self.conversation_history.append({
            "user": message,
            "assistant": result["answer"]
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return result["answer"]
    
    def get_last_result(self) -> Dict[str, Any]:
        """Get metadata from last query."""
        if not self.conversation_history:
            return {"sources": [], "confidence": 0.0}
        
        # This is a simplified version - in practice, you'd store this with each response
        return {"sources": [], "confidence": 0.0}


def create_rag_system_from_config(config_path: str = "Config.yaml") -> RAGSystem:
    """
    Create RAG system from configuration file.
    
    Args:
        config_path: Path to Config.yaml
        
    Returns:
        Configured RAGSystem instance
    """
    # Load config
    config_paths = [
        config_path,
        Path(__file__).parent.parent / config_path,
        Path(__file__).parent.parent.parent / config_path
    ]
    
    config = {}
    for path in config_paths:
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            break
    
    # Extract settings
    paths = config.get('paths', {})
    embedding = config.get('embedding', {})
    doc_processing = config.get('document_processing', {})
    groq_config = config.get('groq', {})
    
    return RAGSystem(
        data_dir=paths.get('data_dir', './data'),
        vectorstore_dir=paths.get('vectorstore_dir', './vectorstore'),
        embedding_model=embedding.get('model_name', 'BAAI/bge-small-en-v1.5'),
        chunk_size=doc_processing.get('chunk_size', 1024),
        chunk_overlap=doc_processing.get('chunk_overlap', 128),
        groq_api_key=groq_config.get('api_key'),
        groq_model=groq_config.get('model', 'llama-3.3-70b-versatile')
    )


"""
Document Indexer - Q1: Mise en place d'un système d'indexation des documents
Handles document loading, splitting, embedding, and storage in vector store.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class DocumentIndexer:
    """
    Q1: Document indexation system with vector store and embeddings.
    Implements the complete pipeline: Loading -> Splitting -> Embedding -> Storage
    """
    
    def __init__(self, 
                 data_dir: str = "./data",
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 collection_name: str = "rag_collection"):
        """
        Initialize the document indexer.
        
        Args:
            data_dir: Directory containing documents to index
            vectorstore_dir: Directory for ChromaDB persistence
            embedding_model: HuggingFace embedding model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
            collection_name: ChromaDB collection name
        """
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        
        # Initialize embedding model (Hugging Face)
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        print(f"✅ Embedding model loaded")
        
        # Initialize ChromaDB vector store
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        
        self.index = None
    
    def load_documents(self) -> List:
        """
        Loading: Load documents using data loader.
        
        Returns:
            List of loaded documents
        """
        print("📂 Loading documents...")
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Use SimpleDirectoryReader for PDF and text files
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            filename_as_id=True
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"   ✅ Loaded {len(documents)} document(s)")
        return documents
    
    def configure_metadata(self, documents: List) -> List:
        """
        Configure document metadata for optimal embedding and retrieval.
        
        Args:
            documents: List of documents to configure
            
        Returns:
            Configured documents with metadata
        """
        print("📝 Configuring document metadata...")
        
        for doc in documents:
            # Set text template for better embedding (includes metadata)
            doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
            
            # Exclude page_label from embeddings (not useful for semantic search)
            if "page_label" not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append("page_label")
        
        print(f"   ✅ Configured {len(documents)} document(s)")
        return documents
    
    def create_text_splitter(self) -> SentenceSplitter:
        """
        Splitting: Create text splitter optimized for document chunks.
        Preserves metadata and optimizes for Markdown format.
        
        Returns:
            Configured SentenceSplitter
        """
        print(f"✂️  Creating text splitter (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        # SentenceSplitter with space separator for better chunk boundaries
        text_splitter = SentenceSplitter(
            separator=" ",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        print(f"   ✅ Text splitter configured")
        return text_splitter
    
    def setup_vector_store(self) -> tuple:
        """
        Storage: Setup ChromaDB vector store for embedding storage.
        
        Returns:
            Tuple of (vector_store, storage_context)
        """
        print(f"💾 Setting up ChromaDB vector store...")
        
        # Create or get ChromaDB collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Create vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"   ✅ ChromaDB collection '{self.collection_name}' ready")
        return vector_store, storage_context
    
    def build_index(self) -> VectorStoreIndex:
        """
        Complete indexation pipeline: Loading -> Splitting -> Embedding -> Storage
        
        Returns:
            Built VectorStoreIndex
        """
        print("\n" + "="*80)
        print("🚀 BUILDING DOCUMENT INDEX")
        print("="*80 + "\n")
        
        # Step 1: Loading - Load documents
        documents = self.load_documents()
        
        # Step 2: Configure metadata
        documents = self.configure_metadata(documents)
        
        # Step 3: Splitting - Create text splitter
        text_splitter = self.create_text_splitter()
        
        # Step 4: Storage - Setup vector store
        vector_store, storage_context = self.setup_vector_store()
        
        # Step 5: Embedding & Storage - Create index with embeddings
        print(f"🔨 Creating vector index with embeddings...")
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[text_splitter],
            show_progress=True
        )
        
        print(f"   ✅ Index created successfully!")
        
        print("\n" + "="*80)
        print("✅ DOCUMENT INDEXATION COMPLETE!")
        print("="*80 + "\n")
        
        return self.index
    
    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing index from vector store.
        
        Returns:
            Loaded VectorStoreIndex or None if not found
        """
        try:
            print(f"📂 Loading existing index from {self.vectorstore_dir}...")
            
            # Get ChromaDB collection
            chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Load index from vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            
            print(f"✅ Index loaded successfully")
            return self.index
            
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            return None
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index, building it if necessary.
        
        Returns:
            VectorStoreIndex instance
        """
        if self.index is None:
            # Try to load existing index first
            if self.load_existing_index() is None:
                # Build new index if loading fails
                self.build_index()
        
        return self.index

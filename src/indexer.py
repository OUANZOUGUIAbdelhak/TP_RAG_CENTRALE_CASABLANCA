"""
Q1: Document Indexation System
===============================
This module handles document indexing for the RAG system.

Pipeline:
1. Loading: Load documents using a data loader
2. Splitting: Divide documents into chunks (optimized for Markdown)
3. Advanced Metadata Extraction: Extract titles and questions answered (using LLM)
4. Embedding: Calculate embeddings using HuggingFace models
5. Storage: Store embeddings + metadata in ChromaDB vector store
"""

import os
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class DocumentIndexer:
    """
    Q1: Document indexing system using ChromaDB and HuggingFace embeddings.
    
    This class implements the complete indexation pipeline:
    - Document loading from a directory
    - Text splitting with metadata preservation
    - Embedding generation using HuggingFace models
    - Storage in ChromaDB vector store
    """
    
    def __init__(self,
                 data_dir: str = "./data",
                 vectorstore_dir: str = "./vectorstore",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 collection_name: str = "rag_collection",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 use_advanced_rag: bool = True):
        """
        Initialize the document indexer.
        
        Args:
            data_dir: Directory containing documents to index
            vectorstore_dir: Directory for ChromaDB persistence
            embedding_model_name: HuggingFace embedding model name
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            collection_name: Name of the ChromaDB collection
            groq_api_key: Groq API key for LLM-based metadata extraction (optional)
            groq_model: Groq model name for metadata extraction
            use_advanced_rag: Whether to use advanced RAG pipeline with title/QA extraction
        """
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.use_advanced_rag = use_advanced_rag
        
        # Initialize embedding model (HuggingFace)
        print(f"🔧 Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        print(f"✅ Embedding model loaded successfully")
        
        # Initialize LLM for advanced metadata extraction (if enabled)
        self.llm = None
        if use_advanced_rag and groq_api_key:
            try:
                print(f"🔧 Initializing Groq LLM for metadata extraction: {groq_model}")
                self.llm = Groq(model=groq_model, api_key=groq_api_key)
                print(f"✅ Groq LLM initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize Groq LLM: {e}")
                print(f"⚠️  Falling back to basic RAG pipeline (no title/QA extraction)")
                self.use_advanced_rag = False
                self.llm = None
        elif use_advanced_rag and not groq_api_key:
            print(f"⚠️  Advanced RAG requested but no Groq API key provided")
            print(f"⚠️  Falling back to basic RAG pipeline (no title/QA extraction)")
            self.use_advanced_rag = False
        
        # Initialize ChromaDB client
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectorstore_dir))
        
        # Store index reference
        self.index = None
    
    def build_index(self) -> VectorStoreIndex:
        """
        Build the vector index from documents in data_dir.
        
        This implements the complete Q1 pipeline with Advanced RAG:
        1. Loading: Load documents with SimpleDirectoryReader
        2. Splitting: Split into chunks with SentenceSplitter
        3. Advanced Metadata Extraction: Extract titles and questions answered (if enabled)
        4. Embedding: Generate embeddings with HuggingFace model
        5. Storage: Store in ChromaDB vector store with enriched metadata
        
        Returns:
            VectorStoreIndex: The built index
        """
        print("\n" + "="*80)
        print("🚀 Q1: BUILDING DOCUMENT INDEX")
        if self.use_advanced_rag:
            print("   🔥 Using Advanced RAG Pipeline (Title + QA Extraction)")
        else:
            print("   📝 Using Basic RAG Pipeline")
        print("="*80 + "\n")
        
        # Step 1: Loading - Load documents from data directory
        print("📂 Step 1: Loading documents...")
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            filename_as_id=True,  # Use filename as document ID
            recursive=True  # Search subdirectories
        ).load_data()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"   ✅ Loaded {len(documents)} document(s)")
        
        # Configure metadata (preserve and store metadata)
        print("\n📝 Step 2: Configuring metadata preservation...")
        for doc in documents:
            # Set text template to include metadata in embeddings
            doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
            # Exclude page_label from embeddings (not useful for semantic search)
            if "page_label" not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append("page_label")
        
        print(f"   ✅ Metadata configured for {len(documents)} document(s)")
        
        # Step 2/3: Splitting + Advanced Metadata Extraction
        print(f"\n✂️  Step 3: Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = SentenceSplitter(
            separator=" ",  # Split on spaces (respects sentence boundaries)
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Build the ingestion pipeline
        if self.use_advanced_rag and self.llm:
            print("   🔥 Setting up Advanced RAG pipeline with metadata extractors...")
            # Create metadata extractors
            title_extractor = TitleExtractor(llm=self.llm, nodes=5)
            qa_extractor = QuestionsAnsweredExtractor(llm=self.llm, questions=3)
            
            # Create ingestion pipeline with all transformations
            pipeline = IngestionPipeline(
                transformations=[text_splitter, title_extractor, qa_extractor]
            )
            
            print("   ✅ Advanced pipeline configured (Title + QA extraction)")
            print("   ⏳ Running pipeline (this may take a while)...")
            
            # Run the pipeline to generate enriched nodes
            nodes = pipeline.run(documents=documents, in_place=True, show_progress=True)
            
            print(f"   ✅ Generated {len(nodes)} enriched nodes with metadata")
            
            # Configure nodes to include metadata in embeddings
            print("   🔧 Configuring nodes to embed metadata...")
            for node in nodes:
                # Set text template to include metadata in embeddings
                # Format: "Title: ...\nQuestions: ...\n\nContent: ..."
                # This ensures title and questions_answered are included in the embedding
                title = node.metadata.get('title', '')
                questions = node.metadata.get('questions_this_excerpt_can_answer', '')
                
                # Build metadata string for embedding
                metadata_parts = []
                if title:
                    metadata_parts.append(f"Title: {title}")
                if questions:
                    if isinstance(questions, list):
                        qa_str = "; ".join(questions)
                    else:
                        qa_str = str(questions)
                    metadata_parts.append(f"Questions this excerpt can answer: {qa_str}")
                
                if metadata_parts:
                    # Set template to include metadata in the embedded text
                    # The template will format: metadata_str + content
                    node.text_template = "{metadata_str}\n\n{content}"
                
                # Ensure only page_label is excluded from embeddings (keep title and questions)
                # Remove title and questions from excluded list if they're there
                excluded = set(node.excluded_embed_metadata_keys or [])
                excluded.discard('title')
                excluded.discard('questions_this_excerpt_can_answer')
                excluded.add('page_label')  # Always exclude page_label
                node.excluded_embed_metadata_keys = list(excluded)
            
            print(f"   ✅ Configured {len(nodes)} nodes to include metadata in embeddings")
            
            # Show sample node with metadata
            if nodes:
                print("\n   🔍 Sample node (with metadata for embedding):")
                sample_node = nodes[0]
                title = sample_node.metadata.get('title', 'N/A')
                questions = sample_node.metadata.get('questions_this_excerpt_can_answer', 'N/A')
                print(f"      - Title: {title}")
                print(f"      - Questions Answered: {questions}")
                print(f"      - Content length: {len(sample_node.text)} chars")
                # Show what will be embedded
                embedded_content = sample_node.get_content(metadata_mode=MetadataMode.LLM)
                print(f"      - Embedded content length: {len(embedded_content)} chars (includes metadata)")
        else:
            # Basic pipeline: just split
            print("   📝 Using basic text splitting (no metadata extraction)")
            nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
            print(f"   ✅ Generated {len(nodes)} nodes")
        
        # Step 4: Setup ChromaDB collection
        print(f"\n💾 Step 4: Setting up ChromaDB vector store...")
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(f"   ✅ ChromaDB collection '{self.collection_name}' ready")
        
        # Step 5: Create index with enriched nodes
        print(f"\n🔨 Step 5: Creating vector index (embedding + storage)...")
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,  # HuggingFace embeddings
            show_progress=True  # Show progress bar
        )
        
        print(f"   ✅ Index created and stored in ChromaDB!")
        
        # Verify metadata storage
        if self.use_advanced_rag and nodes:
            print(f"\n📊 Metadata Summary:")
            nodes_with_title = sum(1 for n in nodes if n.metadata.get('title'))
            nodes_with_qa = sum(1 for n in nodes if n.metadata.get('questions_this_excerpt_can_answer'))
            print(f"   - Nodes with title: {nodes_with_title}/{len(nodes)}")
            print(f"   - Nodes with QA metadata: {nodes_with_qa}/{len(nodes)}")
        
        print("\n" + "="*80)
        print("✅ Q1: INDEX BUILD COMPLETE!")
        print("="*80 + "\n")
        
        return self.index
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index (if built).
        
        Returns:
            VectorStoreIndex or None
        """
        return self.index


def check_data_folder(data_dir: str = "./data") -> bool:
    """
    Check if data folder exists and contains PDF files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if folder exists and contains PDFs, False otherwise
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    pdf_files = list(data_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   - {pdf.name} ({size_mb:.2f} MB)")
    
    return True









print("")

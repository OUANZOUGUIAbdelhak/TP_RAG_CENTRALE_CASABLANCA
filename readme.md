# RAG System - Retrieval Augmented Generation

A comprehensive RAG (Retrieval Augmented Generation) system built with LlamaIndex, featuring document indexing, semantic search, question-answering, and an interactive chatbot interface. The system supports multiple LLM providers with automatic fallback mechanisms.

## 🌟 Features

### Core Capabilities
- **📚 Document Indexing (Q1)**: Advanced document processing with metadata extraction
  - Support for PDF, DOCX, and Markdown files
  - Intelligent text chunking with configurable overlap
  - Advanced RAG pipeline with title and Q&A extraction
  - ChromaDB vector store for efficient storage

- **🔍 Semantic Search (Q2)**: Powerful document retrieval
  - Vector similarity search with HuggingFace embeddings
  - Configurable top-k results
  - Similarity scoring and ranking
  - Metadata filtering support

- **💬 Question Answering (Q3)**: LLM-powered Q&A system
  - Context-aware answer generation
  - Source citation and confidence scoring
  - Custom prompt templates optimized for RAG
  - Document-specific filtering

- **📊 Evaluation (Q4)**: System performance assessment
  - Relevance metrics
  - Accuracy evaluation
  - Quality checks

- **🤖 Interactive Chatbot (Q5)**: Conversational interface
  - Multi-session support
  - Conversation history management
  - Context-aware responses
  - Session persistence

### Advanced Features
- **🔄 LLM Fallback System**: Automatic failover between Groq and Gemini
- **🌐 RESTful API**: FastAPI backend with Swagger documentation
- **💻 Modern Frontend**: React-based web interface
- **⚙️ Flexible Configuration**: YAML-based configuration system
- **📁 File Management**: Upload, organize, and manage documents via API

## 🏗️ Architecture

The system follows a modular architecture with five core modules:

```
RAG System Architecture
├── Q1: Document Indexer (indexer.py)
│   ├── Document loading and parsing
│   ├── Text chunking with overlap
│   ├── Advanced metadata extraction (titles, Q&A)
│   └── Vector embedding and storage
│
├── Q2: Document Retriever (retriever.py)
│   ├── Vector similarity search
│   ├── Top-k retrieval
│   └── Similarity scoring
│
├── Q3: QA System (qa_system.py)
│   ├── Context retrieval
│   ├── LLM-based answer generation
│   └── Source citation
│
├── Q4: Evaluator (evaluator.py)
│   ├── Relevance metrics
│   └── Accuracy assessment
│
└── Q5: Chatbot (chatbot.py)
    ├── Conversation management
    ├── Session handling
    └── Context building
```

### Technology Stack

**Backend:**
- Python 3.9+
- LlamaIndex (vector store and RAG framework)
- ChromaDB (vector database)
- FastAPI (REST API)
- HuggingFace (embeddings)
- Groq & Gemini (LLM providers)

**Frontend:**
- React + Vite
- Tailwind CSS
- Axios (API client)

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Node.js 16+ (for frontend)
- pip package manager

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TP_RAG_CENTRALE_CASABLANCA
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the system**
   Edit `Config.yaml` and add your API keys:
   ```yaml
   groq:
     api_key: "your_groq_api_key"
   
   gemini:
     api_key: "your_gemini_api_key"
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

## ⚙️ Configuration

The system is configured via `Config.yaml`. Key settings:

### Paths
```yaml
paths:
  data_dir: "./data/files"          # Documents to index
  vectorstore_dir: "./data/vectorstore"  # Vector database
  chat_sessions_dir: "./data/chat_sessions"  # Chat history
```

### Embedding Model
```yaml
embedding:
  model_name: "BAAI/bge-large-en-v1.5"  # HuggingFace model
```

### Document Processing
```yaml
document_processing:
  chunk_size: 1024      # Characters per chunk
  chunk_overlap: 128    # Overlap between chunks
```

### LLM Settings
```yaml
groq:
  api_key: "your_key"
  model: "llama-3.3-70b-versatile"
  temperature: 0.7

gemini:
  api_key: "your_key"
  model: "gemini-2.0-flash"
  temperature: 0.7
```

## 🚀 Usage

### Command Line Interface

The system provides a CLI for all operations:

#### 1. Build Index (Q1)
```bash
python Cli.py build
```
This will:
- Load documents from `data/files/`
- Process and chunk documents
- Extract metadata (if advanced RAG enabled)
- Generate embeddings
- Store in ChromaDB

#### 2. Search Documents (Q2)
```bash
python Cli.py search "your query" -k 10
```
Returns top-k most relevant documents with similarity scores.

#### 3. Ask Questions (Q3)
```bash
python Cli.py ask "What is machine learning?"
```
Generates an answer using retrieved context and LLM.

#### 4. Evaluate System (Q4)
```bash
python Cli.py evaluate --quick
```
Runs quality checks and performance metrics.

#### 5. Interactive Chat (Q5)
```bash
python Cli.py chat
```
Starts an interactive chatbot session.

### REST API

Start the API server:
```bash
# From project root
python -m src.backend.api

# Or from src/backend
cd src/backend
python api.py
```

The API will be available at `http://127.0.0.1:8000`

#### Key Endpoints

**File Management:**
- `GET /api/files` - List all files and folders
- `POST /api/upload` - Upload a document
- `POST /api/create-folder` - Create a folder
- `DELETE /api/delete?path=<file_path>` - Delete file/folder

**RAG Operations:**
- `POST /api/build-index` - Build or rebuild the index
- `POST /api/search` - Search documents
  ```json
  {
    "query": "your search query",
    "k": 10
  }
  ```

**Chat:**
- `POST /api/chat` - Send a chat message
  ```json
  {
    "message": "your question",
    "session_id": "optional_session_id"
  }
  ```
- `GET /api/sessions` - List all chat sessions
- `POST /api/session/new` - Create a new session
- `GET /api/session/{id}` - Get session history

**Interactive API Documentation:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Frontend Usage

1. **Start the backend API** (see above)

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the web interface**
   - Open http://localhost:5173 (or the port shown in terminal)

4. **Use the interface**
   - Upload documents via the file explorer
   - Build the index
   - Search documents
   - Chat with the AI assistant

## 🔄 LLM Fallback Mechanism

The system includes an intelligent fallback system for LLM providers:

1. **Primary**: Attempts to use Groq API
2. **Fallback**: Automatically switches to Gemini if Groq fails
3. **Transparent**: No code changes needed - works automatically

The fallback is configured in `Config.yaml` and works across all modules (Q1, Q3, Q5).

## 📁 Project Structure

```
TP_RAG_CENTRALE_CASABLANCA/
├── Config.yaml                 # Main configuration file
├── Cli.py                      # Command-line interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Data directory
│   ├── files/                  # Documents to index
│   ├── vectorstore/            # ChromaDB storage
│   └── chat_sessions/         # Chat history
│
├── src/                        # Source code
│   ├── backend/
│   │   ├── api.py             # FastAPI backend
│   │   └── README.md          # Backend documentation
│   │
│   ├── indexer.py             # Q1: Document indexing
│   ├── retriever.py           # Q2: Document retrieval
│   ├── qa_system.py           # Q3: Question answering
│   ├── evaluator.py           # Q4: System evaluation
│   ├── chatbot.py             # Q5: Interactive chatbot
│   ├── rag_system.py          # Complete RAG system
│   └── llm_fallback.py        # LLM fallback mechanism
│
└── frontend/                   # React frontend
    ├── src/
    │   ├── components/        # React components
    │   ├── services/         # API client
    │   └── App.jsx           # Main app component
    ├── package.json
    └── vite.config.js
```

## 🔧 Development

### Running Tests

```bash
# Test CLI commands
python Cli.py build
python Cli.py search "test query"

# Test API endpoints
curl http://127.0.0.1:8000/api/health
```

### Adding New Features

1. **New LLM Provider**: Extend `llm_fallback.py`
2. **New Document Type**: Update `indexer.py` document loader
3. **New API Endpoint**: Add to `src/backend/api.py`
4. **Frontend Component**: Add to `frontend/src/components/`

## 📝 API Examples

### Upload a Document
```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@document.pdf"
```

### Build Index
```bash
curl -X POST http://127.0.0.1:8000/api/build-index
```

### Search Documents
```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 5}'
```

### Chat
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "session_id": "session_1"}'
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

**API Key Issues:**
- Verify API keys in `Config.yaml`
- Check API key permissions and quotas
- The system will automatically fallback to Gemini if Groq fails

**Index Not Found:**
- Build the index first: `python Cli.py build` or `POST /api/build-index`
- Ensure documents exist in `data/files/`

**Port Already in Use:**
- Change port in `src/backend/api.py` or kill the process using port 8000

## 📄 License

This project is part of an academic assignment.

## 👥 Team

- OUANZOUGUI Abdelhak
- BELLMIR Omar
- BOURHAIM Ayoub
- DAHHASSI Chaymae
- AIT BIHI Laila
- EL ABDI Ibrahim

## 🙏 Acknowledgments

- LlamaIndex for the RAG framework
- ChromaDB for vector storage
- HuggingFace for embedding models
- Groq and Google for LLM APIs

## 📚 Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)

---

**Happy RAG-ing! 🚀**

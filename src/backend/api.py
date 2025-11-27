"""
FastAPI Backend for RAG System
Handles document management, RAG operations, and chat sessions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
import shutil
import uuid
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path (we're in src/backend/, so go up one level to src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the new modular RAG system
from rag_system import RAGSystem, create_rag_system_from_config
from Chatbot import RAGChatbot

# Initialize FastAPI
app = FastAPI(
    title="RAG System API",
    description="Backend API for Retrieval Augmented Generation System",
    version="1.0.0"
)

# CORS middleware for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
def load_config():
    """Load configuration from Config.yaml"""
    import yaml
    config_paths = [
        "Config.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "Config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "Config.yaml"),
    ]
    
    for path in config_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except:
                continue
    
    return {}

CONFIG = load_config()

# Global configuration
DATA_DIR = Path(CONFIG.get('paths', {}).get('data_dir', './data'))
VECTORSTORE_DIR = Path(CONFIG.get('paths', {}).get('vectorstore_dir', './vectorstore'))
SESSIONS_DIR = Path("./chat_sessions")
EMBEDDING_MODEL = CONFIG.get('embedding', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Global instances
rag_system = None
chatbot_instances = {}  # session_id -> chatbot instance

# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    message: str
    session_id: str
    document_path: Optional[str] = None  # Specific document to query

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    session_id: str
    timestamp: str

class FileInfo(BaseModel):
    name: str
    path: str
    type: str  # 'file' or 'folder'
    size: Optional[int] = None
    modified: Optional[str] = None

class BuildIndexRequest(BaseModel):
    folder_path: Optional[str] = None  # If None, index all data

class SessionInfo(BaseModel):
    session_id: str
    name: str
    created_at: str
    message_count: int

# ============================================================================
# Helper Functions
# ============================================================================

def get_file_tree(base_path: Path = DATA_DIR) -> List[FileInfo]:
    """Get file tree structure"""
    items = []
    
    try:
        for item in sorted(base_path.iterdir()):
            if item.name.startswith('.'):
                continue
            
            if item.is_file():
                # Only include supported formats
                if item.suffix.lower() in ['.pdf', '.docx', '.md']:
                    items.append(FileInfo(
                        name=item.name,
                        path=str(item.relative_to(DATA_DIR)),
                        type='file',
                        size=item.stat().st_size,
                        modified=datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    ))
            elif item.is_dir():
                items.append(FileInfo(
                    name=item.name,
                    path=str(item.relative_to(DATA_DIR)),
                    type='folder'
                ))
    except Exception as e:
        print(f"Error reading directory: {e}")
    
    return items

def get_all_documents() -> List[Path]:
    """Get all supported documents recursively"""
    docs = []
    for ext in ['.pdf', '.docx', '.md']:
        docs.extend(DATA_DIR.rglob(f'*{ext}'))
    return docs

def initialize_rag_system():
    """Initialize or reinitialize RAG system"""
    global rag_system
    
    if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
        return False
    
    try:
        rag_system = create_rag_system_from_config("Config.yaml")
        # Initialize components and try to load existing index
        rag_system.initialize_components()
        if rag_system.load_index():
            return True
        return False
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

def get_or_create_chatbot(session_id: str) -> RAGChatbot:
    """Get existing chatbot or create new one for session"""
    global rag_system
    
    # Ensure RAG system is initialized
    if not rag_system:
        if not initialize_rag_system():
            raise HTTPException(
                status_code=400,
                detail="Index not built. Build index first."
            )
    
    # Ensure QA system is available for chatbot
    if not rag_system.qa_system:
        raise HTTPException(
            status_code=400,
            detail="QA system not initialized. Build index first."
        )
    
    if session_id not in chatbot_instances:
        chatbot_instances[session_id] = RAGChatbot(
            qa_system=rag_system.qa_system,
            max_history=10
        )
    return chatbot_instances[session_id]

def load_session_history(session_id: str) -> List[Dict]:
    """Load chat history for a session"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_session_history(session_id: str, history: List[Dict]):
    """Save chat history for a session"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "RAG System API is running",
        "version": "1.0.0"
    }

@app.get("/api/health")
async def health_check():
    """Check system health"""
    index_exists = VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir())
    doc_count = len(get_all_documents())
    
    return {
        "status": "healthy",
        "index_built": index_exists,
        "document_count": doc_count,
        "rag_system_ready": rag_system is not None
    }

# ============================================================================
# File Management Endpoints
# ============================================================================

@app.get("/api/files")
async def list_files(path: str = ""):
    """List files and folders in a directory"""
    try:
        target_path = DATA_DIR / path if path else DATA_DIR
        
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        
        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        items = get_file_tree(target_path)
        return {"items": items, "current_path": path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    folder: str = ""
):
    """Upload a file to a specific folder"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.docx', '.md']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Only PDF, DOCX, and MD are supported."
            )
        
        # Create target directory
        target_dir = DATA_DIR / folder if folder else DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = target_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "path": str(file_path.relative_to(DATA_DIR)),
            "size": len(content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/create-folder")
async def create_folder(path: str = Body(..., embed=True)):
    """Create a new folder"""
    try:
        folder_path = DATA_DIR / path
        folder_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "status": "success",
            "message": f"Folder created: {path}",
            "path": path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete")
async def delete_item(path: str):
    """Delete a file or folder"""
    try:
        target_path = DATA_DIR / path
        
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Item not found")
        
        if target_path.is_file():
            target_path.unlink()
            return {"status": "success", "message": f"File deleted: {path}"}
        elif target_path.is_dir():
            shutil.rmtree(target_path)
            return {"status": "success", "message": f"Folder deleted: {path}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/file-content")
async def get_file_content(path: str):
    """Get file content (for markdown preview)"""
    try:
        file_path = DATA_DIR / path
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # For markdown files, return content
        if file_path.suffix.lower() == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content, "type": "markdown"}
        
        # For PDFs and DOCX, return file info (path will be used to fetch via /api/file endpoint)
        return {"path": str(file_path.relative_to(DATA_DIR)), "type": file_path.suffix.lower()[1:]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/file")
async def serve_file(path: str):
    """Serve file directly (for PDF, DOCX, etc.)"""
    try:
        file_path = DATA_DIR / path
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        media_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.md': 'text/markdown'
        }
        
        media_type = media_type_map.get(file_path.suffix.lower(), 'application/octet-stream')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RAG Endpoints
# ============================================================================

@app.post("/api/build-index")
async def build_index(request: BuildIndexRequest = None):
    """Build or rebuild the document index"""
    try:
        docs = get_all_documents()
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No documents found. Upload documents first."
            )
        
        # Build index using the new modular RAG system
        global rag_system
        
        rag_system = create_rag_system_from_config("Config.yaml")
        rag_system.initialize_components()
        rag_system.build_index()
        
        return {
            "status": "success",
            "message": f"Index built successfully with {len(docs)} documents",
            "document_count": len(docs),
            "progress": 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/vectorstore")
async def delete_vectorstore():
    """Delete the entire vectorstore to start fresh"""
    try:
        import shutil
        import time
        import gc
        
        # First, close all ChromaDB connections by clearing global instances
        global rag_system, chatbot_instances
        
        # Clear chatbot instances
        chatbot_instances = {}
        rag_system = None
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Small delay to let file handles release (especially on Windows)
        time.sleep(0.5)
        
        if VECTORSTORE_DIR.exists():
            # Try to delete files individually first (helps on Windows)
            try:
                # Delete all files in the directory first
                for file_path in VECTORSTORE_DIR.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                        except PermissionError:
                            # File might be locked, try again after a short delay
                            time.sleep(0.2)
                            file_path.unlink()
            except Exception as e:
                print(f"Warning: Error deleting individual files: {e}")
            
            # Now delete the directory structure
            try:
                shutil.rmtree(VECTORSTORE_DIR, ignore_errors=False)
            except PermissionError as pe:
                # On Windows, files might still be locked
                # Try one more time after a longer delay
                time.sleep(1)
                gc.collect()
                try:
                    shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
                except:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not delete vectorstore. Files may be locked. Error: {str(pe)}. Try closing any applications using the vectorstore and try again."
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting vectorstore: {str(e)}"
                )
            
            # Recreate the directory
            VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)
            
            return {
                "status": "success",
                "message": "Vectorstore deleted successfully. You can rebuild the index now."
            }
        else:
            # Directory doesn't exist, create it
            VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)
            return {
                "status": "success",
                "message": "Vectorstore was already empty."
            }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error deleting vectorstore: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete vectorstore: {str(e)}. Check server logs for details."
        )

@app.post("/api/search")
async def search_documents(
    query: str = Body(..., embed=True),
    k: int = Body(5, embed=True)
):
    """Search for relevant documents"""
    try:
        if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
            raise HTTPException(
                status_code=400,
                detail="Index not built. Build index first."
            )
        
        global rag_system
        if not rag_system:
            rag_system = create_rag_system_from_config("Config.yaml")
            if not rag_system.load_index():
                raise HTTPException(
                    status_code=400,
                    detail="Index not built. Build index first."
                )
        
        results = rag_system.search_documents(query, k=k)
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat interaction"""
    try:
        # Check if index is built
        if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
            raise HTTPException(
                status_code=400,
                detail="Index not built. Build index first."
            )
        
        # Get or create chatbot for this session
        chatbot = get_or_create_chatbot(message.session_id)
        
        # If specific document is provided, search only in that document
        if message.document_path:
            # TODO: Implement document-specific search
            # For now, use regular chat
            pass
        
        # Get response from chatbot
        answer = chatbot.chat(message.message, verbose=False)
        
        # Query again to get sources and confidence
        result = chatbot.rag_system.query(message.message)
        confidence = result.get('confidence', 0.0)
        sources = result.get('sources', [])
        
        # Load session history
        history = load_session_history(message.session_id)
        
        # Add to history
        history.append({
            "role": "user",
            "content": message.message,
            "timestamp": datetime.now().isoformat()
        })
        history.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save session history
        save_session_history(message.session_id, history)
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            session_id=message.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions"""
    try:
        sessions = []
        
        for session_file in SESSIONS_DIR.glob("*.json"):
            session_id = session_file.stem
            history = load_session_history(session_id)
            
            # Get session name (first user message or default)
            name = "New Chat"
            if history:
                for msg in history:
                    if msg.get('role') == 'user':
                        name = msg.get('content', '')[:50]
                        break
            
            sessions.append(SessionInfo(
                session_id=session_id,
                name=name,
                created_at=history[0]['timestamp'] if history else datetime.now().isoformat(),
                message_count=len([m for m in history if m.get('role') == 'user'])
            ))
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get chat history for a specific session"""
    try:
        history = load_session_history(session_id)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/new")
async def create_session():
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        save_session_history(session_id, [])
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "New session created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
        
        # Remove chatbot instance
        if session_id in chatbot_instances:
            del chatbot_instances[session_id]
        
        return {
            "status": "success",
            "message": f"Session {session_id} deleted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Initialize on startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🚀 Starting RAG System API...")
    print(f"📁 Data directory: {DATA_DIR.absolute()}")
    print(f"🗄️  Vector store: {VECTORSTORE_DIR.absolute()}")
    
    # Try to initialize RAG system if index exists
    if initialize_rag_system():
        print("✅ RAG System initialized")
    else:
        print("⚠️  RAG System not initialized - build index first")
    
    print("✅ API Server ready!")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 RAG System Backend API")
    print("="*60)
    print("\n📡 Starting server at: http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/docs")
    print("\n💡 Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


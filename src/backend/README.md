# FastAPI Backend - RAG System

RESTful API backend for the RAG system.

## Quick Start

```bash
python backend/api.py
```

Server starts at: **http://127.0.0.1:8000**

## API Documentation

Interactive API docs available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Endpoints

### Health
- `GET /api/health` - System health check

### Files
- `GET /api/files` - List files and folders
- `POST /api/upload` - Upload file
- `POST /api/create-folder` - Create folder
- `DELETE /api/delete` - Delete file/folder
- `GET /api/file-content` - Get file content

### RAG
- `POST /api/build-index` - Build document index
- `POST /api/search` - Search documents

### Chat
- `POST /api/chat` - Send chat message
- `GET /api/sessions` - List chat sessions
- `GET /api/session/{id}` - Get session history
- `POST /api/session/new` - Create new session
- `DELETE /api/session/{id}` - Delete session

## Configuration

- **Host**: 127.0.0.1
- **Port**: 8000
- **CORS**: Enabled for localhost:3000 (React)
- **Data Directory**: ./data
- **Vector Store**: ./vectorstore
- **Sessions**: ./chat_sessions

## Tech Stack

- FastAPI - Modern Python API framework
- Uvicorn - ASGI server
- Pydantic - Data validation
- CORS Middleware - Cross-origin requests

## Requirements

- Python 3.9+
- See requirements.txt in project root

## Development

```bash
# Run with auto-reload
uvicorn backend.api:app --reload

# Run on custom port
python backend/api.py --port 8001
```

## Production

```bash
# Use gunicorn with uvicorn workers
gunicorn backend.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Environment Variables

Optional:
- `HUGGINGFACEHUB_API_TOKEN` - For LLM access

## Features

- File upload/download
- Folder management
- RAG indexing
- Document search
- Chat with AI
- Session management
- Source references
- Confidence scores

## Architecture

```
FastAPI Backend
├── File Management
│   ├── Upload (multipart/form-data)
│   ├── List (directory tree)
│   └── Delete (files/folders)
├── RAG System
│   ├── Build Index (vectorization)
│   └── Search (similarity search)
└── Chat System
    ├── Send Message (with context)
    ├── Sessions (multiple conversations)
    └── History (persistent storage)
```

## CORS

Configured to allow:
- http://localhost:3000 (Vite dev server)
- http://localhost:5173 (Alternative Vite port)

## Data Persistence

- **Files**: ./data directory
- **Vector DB**: ./vectorstore directory
- **Chat History**: ./chat_sessions/*.json

## Error Handling

All endpoints return structured errors:
```json
{
  "detail": "Error message"
}
```

HTTP status codes:
- 200: Success
- 400: Bad request
- 404: Not found
- 500: Server error

## Testing

Use the interactive docs:
```
http://127.0.0.1:8000/docs
```

Or with curl:
```bash
# Health check
curl http://127.0.0.1:8000/api/health

# List files
curl http://127.0.0.1:8000/api/files

# Upload file
curl -X POST -F "file=@document.pdf" http://127.0.0.1:8000/api/upload
```

## Logging

Server logs all requests to console.

## Security

- CORS restricted to localhost
- File type validation on upload
- Path sanitization for file access
- No authentication (development only)

**Note:** Add authentication for production!


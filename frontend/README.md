# React Frontend - RAG System

Beautiful ChatGPT-style interface for the RAG system.

## Quick Start

```bash
# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Features

- 📁 File Explorer with drag & drop upload
- 📄 Document Viewer (PDF, DOCX, MD)
- 💬 Chat Interface with AI
- 🎯 Drag documents to chat for specific queries
- 🔄 Multiple chat sessions
- 🎨 Beautiful ChatGPT-style design

## Tech Stack

- React 18
- Vite
- TailwindCSS
- Axios
- React Dropzone
- React Markdown
- React PDF
- Lucide React

## Development

```bash
npm run dev    # Start dev server
npm run build  # Build for production
npm run preview # Preview production build
```

## Configuration

- **Port**: 3000 (configurable in vite.config.js)
- **API Proxy**: Forwards /api to http://127.0.0.1:8000

## Requirements

- Node.js 18+
- npm or yarn

## Backend

Make sure the backend is running at http://127.0.0.1:8000

```bash
# In project root
python backend/api.py
```

## Documentation

See parent directory for full documentation:
- REACT_SETUP.md - Complete setup guide
- REACT_QUICK_START.md - Quick start guide


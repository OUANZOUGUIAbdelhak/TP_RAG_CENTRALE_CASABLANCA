import { useState, useEffect } from 'react'
import FileExplorer from './components/FileExplorer'
import DocumentViewer from './components/DocumentViewer'
import ChatSidebar from './components/ChatSidebar'
import Header from './components/Header'
import api from './services/api'

function App() {
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [indexStatus, setIndexStatus] = useState(false)
  const [currentSession, setCurrentSession] = useState(null)
  const [draggedDocument, setDraggedDocument] = useState(null)
  const [buildingIndex, setBuildingIndex] = useState(false)
  const [buildProgress, setBuildProgress] = useState({ message: '', progress: 0 })

  // Check system health on mount
  useEffect(() => {
    checkHealth()
    createInitialSession()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await api.get('/health')
      setIndexStatus(response.data.index_built)
    } catch (error) {
      console.error('Health check failed:', error)
    }
  }

  const createInitialSession = async () => {
    try {
      const response = await api.post('/session/new')
      setCurrentSession(response.data.session_id)
    } catch (error) {
      console.error('Failed to create session:', error)
    }
  }

  const handleBuildIndex = async () => {
    setBuildingIndex(true)
    setBuildProgress({ message: 'Starting indexation...', progress: 0 })
    
    try {
      // Show progress updates
      setBuildProgress({ message: 'Loading documents...', progress: 25 })
      
      await new Promise(resolve => setTimeout(resolve, 500))
      setBuildProgress({ message: 'Splitting documents into chunks...', progress: 50 })
      
      await new Promise(resolve => setTimeout(resolve, 500))
      setBuildProgress({ message: 'Creating embeddings (this may take a minute)...', progress: 75 })
      
      const response = await api.post('/build-index')
      
      setBuildProgress({ message: 'Finalizing...', progress: 100 })
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setIndexStatus(true)
      setBuildingIndex(false)
      setBuildProgress({ message: '', progress: 0 })
      
      alert(`Index built successfully! ${response.data.document_count} document(s) indexed.`)
    } catch (error) {
      console.error('Failed to build index:', error)
      setBuildingIndex(false)
      setBuildProgress({ message: '', progress: 0 })
      alert('Failed to build index. Make sure documents are uploaded.')
    }
  }

  const handleRefresh = () => {
    checkHealth()
  }

  const handleDeleteVectorstore = async () => {
    if (!confirm('Are you sure you want to delete all embeddings? You will need to rebuild the index.')) {
      return
    }

    try {
      const response = await api.delete('/vectorstore')
      setIndexStatus(false)
      alert(response.data?.message || 'Vectorstore deleted successfully! Please rebuild the index.')
    } catch (error) {
      console.error('Failed to delete vectorstore:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to delete vectorstore.'
      
      // Provide helpful error message
      if (errorMessage.includes('locked') || errorMessage.includes('Permission')) {
        alert(`Failed to delete vectorstore: Files may be locked.\n\nPlease try:\n1. Stop the backend server\n2. Manually delete the 'vectorstore' folder\n3. Restart the server\n\nError: ${errorMessage}`)
      } else {
        alert(`Failed to delete vectorstore: ${errorMessage}`)
      }
    }
  }

  const handleNewSession = async () => {
    try {
      const response = await api.post('/session/new')
      setCurrentSession(response.data.session_id)
    } catch (error) {
      console.error('Failed to create new session:', error)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-chatgpt-50">
      {/* Header */}
      <Header 
        indexStatus={indexStatus}
        onBuildIndex={handleBuildIndex}
        onRefresh={handleRefresh}
        buildingIndex={buildingIndex}
        buildProgress={buildProgress}
        onDeleteVectorstore={handleDeleteVectorstore}
      />

      {/* Main 3-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - File Explorer */}
        <div className="w-80 border-r border-chatgpt-200 bg-white">
          <FileExplorer 
            onSelectDocument={setSelectedDocument}
            selectedDocument={selectedDocument}
            onDragStart={setDraggedDocument}
          />
        </div>

        {/* Middle - Document Viewer */}
        <div className="flex-1 bg-chatgpt-50 overflow-auto">
          <DocumentViewer document={selectedDocument} />
        </div>

        {/* Right Sidebar - Chat */}
        <div className="w-96 border-l border-chatgpt-200 bg-white">
          <ChatSidebar 
            sessionId={currentSession}
            onNewSession={handleNewSession}
            draggedDocument={draggedDocument}
            onDragEnd={() => setDraggedDocument(null)}
            indexStatus={indexStatus}
          />
        </div>
      </div>
    </div>
  )
}

export default App


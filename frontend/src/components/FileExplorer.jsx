import { useState, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { 
  Upload, 
  File, 
  Folder, 
  FolderPlus, 
  Trash2, 
  FileText, 
  FileCode,
  ChevronRight,
  ChevronDown
} from 'lucide-react'
import api from '../services/api'

function FileExplorer({ onSelectDocument, selectedDocument, onDragStart }) {
  const [files, setFiles] = useState([])
  const [currentPath, setCurrentPath] = useState('')
  const [expandedFolders, setExpandedFolders] = useState(new Set())
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    loadFiles()
  }, [currentPath])

  const loadFiles = async () => {
    try {
      const response = await api.get('/files', { params: { path: currentPath } })
      setFiles(response.data.items)
    } catch (error) {
      console.error('Failed to load files:', error)
    }
  }

  const onDrop = useCallback(async (acceptedFiles) => {
    setUploading(true)
    
    for (const file of acceptedFiles) {
      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('folder', currentPath)
        
        await api.post('/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      } catch (error) {
        console.error(`Failed to upload ${file.name}:`, error)
      }
    }
    
    setUploading(false)
    loadFiles()
  }, [currentPath])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/markdown': ['.md']
    },
    multiple: true
  })

  const handleCreateFolder = async () => {
    const folderName = prompt('Enter folder name:')
    if (!folderName) return

    try {
      const newPath = currentPath ? `${currentPath}/${folderName}` : folderName
      await api.post('/create-folder', { path: newPath })
      loadFiles()
    } catch (error) {
      console.error('Failed to create folder:', error)
    }
  }

  const handleDelete = async (item) => {
    if (!confirm(`Delete ${item.name}?`)) return

    try {
      await api.delete('/delete', { params: { path: item.path } })
      loadFiles()
      if (selectedDocument?.path === item.path) {
        onSelectDocument(null)
      }
    } catch (error) {
      console.error('Failed to delete:', error)
    }
  }

  const handleFolderClick = (folder) => {
    const newExpanded = new Set(expandedFolders)
    if (newExpanded.has(folder.path)) {
      newExpanded.delete(folder.path)
    } else {
      newExpanded.add(folder.path)
    }
    setExpandedFolders(newExpanded)
  }

  const handleFileClick = (file) => {
    onSelectDocument(file)
  }

  const handleFileDragStart = (file, e) => {
    e.dataTransfer.effectAllowed = 'copy'
    onDragStart(file)
  }

  const getFileIcon = (filename) => {
    if (filename.endsWith('.pdf')) {
      return <File className="w-4 h-4 text-red-500" />
    } else if (filename.endsWith('.docx')) {
      return <FileText className="w-4 h-4 text-blue-500" />
    } else if (filename.endsWith('.md')) {
      return <FileCode className="w-4 h-4 text-purple-500" />
    }
    return <File className="w-4 h-4 text-chatgpt-500" />
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-chatgpt-200">
        <h2 className="text-lg font-semibold text-chatgpt-900 mb-3">Documents</h2>
        
        {/* Upload Area */}
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-4 text-center cursor-pointer
            transition-colors
            ${isDragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-chatgpt-300 hover:border-chatgpt-400 hover:bg-chatgpt-50'
            }
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-8 h-8 mx-auto mb-2 text-chatgpt-500" />
          <p className="text-sm text-chatgpt-600">
            {uploading ? 'Uploading...' : 'Drop files or click to upload'}
          </p>
          <p className="text-xs text-chatgpt-500 mt-1">
            PDF, DOCX, MD
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-2 mt-3">
          <button
            onClick={handleCreateFolder}
            className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-chatgpt-100 hover:bg-chatgpt-200 rounded-lg transition-colors"
          >
            <FolderPlus className="w-4 h-4 text-chatgpt-700" />
            <span className="text-sm text-chatgpt-700">New Folder</span>
          </button>
        </div>
      </div>

      {/* File List */}
      <div className="flex-1 overflow-y-auto p-2">
        {files.length === 0 ? (
          <div className="text-center py-8">
            <Folder className="w-12 h-12 mx-auto mb-2 text-chatgpt-300" />
            <p className="text-sm text-chatgpt-500">No files yet</p>
            <p className="text-xs text-chatgpt-400 mt-1">Upload documents to get started</p>
          </div>
        ) : (
          <div className="space-y-1">
            {files.map((item) => (
              <div key={item.path}>
                {item.type === 'folder' ? (
                  // Folder Item
                  <div className="group">
                    <div
                      onClick={() => handleFolderClick(item)}
                      className="flex items-center justify-between p-2 rounded-lg hover:bg-chatgpt-100 cursor-pointer"
                    >
                      <div className="flex items-center space-x-2 flex-1">
                        {expandedFolders.has(item.path) ? (
                          <ChevronDown className="w-4 h-4 text-chatgpt-500" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-chatgpt-500" />
                        )}
                        <Folder className="w-4 h-4 text-yellow-500" />
                        <span className="text-sm text-chatgpt-900">{item.name}</span>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDelete(item)
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                      >
                        <Trash2 className="w-3 h-3 text-red-500" />
                      </button>
                    </div>
                  </div>
                ) : (
                  // File Item
                  <div
                    draggable
                    onDragStart={(e) => handleFileDragStart(item, e)}
                    onClick={() => handleFileClick(item)}
                    className={`
                      group flex items-center justify-between p-2 rounded-lg cursor-pointer
                      ${selectedDocument?.path === item.path 
                        ? 'bg-blue-100 border-blue-300' 
                        : 'hover:bg-chatgpt-100'
                      }
                    `}
                  >
                    <div className="flex items-center space-x-2 flex-1 min-w-0">
                      {getFileIcon(item.name)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-chatgpt-900 truncate">{item.name}</p>
                        <p className="text-xs text-chatgpt-500">
                          {formatFileSize(item.size)}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDelete(item)
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                    >
                      <Trash2 className="w-3 h-3 text-red-500" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="p-3 border-t border-chatgpt-200 bg-chatgpt-50">
        <p className="text-xs text-chatgpt-600">
          {files.filter(f => f.type === 'file').length} file(s) • {files.filter(f => f.type === 'folder').length} folder(s)
        </p>
      </div>
    </div>
  )
}

export default FileExplorer


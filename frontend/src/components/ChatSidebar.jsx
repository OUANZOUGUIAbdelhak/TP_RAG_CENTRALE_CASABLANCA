import { useState, useEffect, useRef } from 'react'
import { 
  Send, 
  Plus, 
  MessageSquare, 
  Trash2, 
  FileText,
  X,
  Link2
} from 'lucide-react'
import api from '../services/api'

function ChatSidebar({ sessionId, onNewSession, draggedDocument, onDragEnd, indexStatus }) {
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [sending, setSending] = useState(false)
  const [sessions, setSessions] = useState([])
  const [showSessions, setShowSessions] = useState(false)
  const [attachedDocument, setAttachedDocument] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const messagesEndRef = useRef(null)
  const chatAreaRef = useRef(null)

  useEffect(() => {
    if (sessionId) {
      loadSessionHistory()
      loadSessions()
    }
  }, [sessionId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadSessionHistory = async () => {
    try {
      const response = await api.get(`/session/${sessionId}`)
      setMessages(response.data.history)
    } catch (error) {
      console.error('Failed to load session history:', error)
    }
  }

  const loadSessions = async () => {
    try {
      const response = await api.get('/sessions')
      setSessions(response.data.sessions)
    } catch (error) {
      console.error('Failed to load sessions:', error)
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || sending || !indexStatus) return

    const userMessage = inputMessage.trim()
    setInputMessage('')
    setSending(true)

    // Add user message immediately for better UX
    const tempUserMsg = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, tempUserMsg])

    try {
      const response = await api.post('/chat', {
        message: userMessage,
        session_id: sessionId,
        document_path: attachedDocument?.path || null
      })

      // Add assistant message
      const assistantMsg = {
        role: 'assistant',
        content: response.data.answer,
        confidence: response.data.confidence,
        sources: response.data.sources,
        timestamp: response.data.timestamp
      }
      setMessages(prev => [...prev, assistantMsg])

      // Clear attached document after sending
      if (attachedDocument) {
        setAttachedDocument(null)
      }

      // Reload sessions to update counts
      loadSessions()
    } catch (error) {
      console.error('Failed to send message:', error)
      // Remove temp user message on error
      setMessages(prev => prev.slice(0, -1))
      alert('Failed to send message. Please try again.')
    } finally {
      setSending(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleNewSession = async () => {
    await onNewSession()
    setMessages([])
    setAttachedDocument(null)
    loadSessions()
  }

  const handleSwitchSession = async (session) => {
    try {
      const response = await api.get(`/session/${session.session_id}`)
      setMessages(response.data.history)
      setShowSessions(false)
    } catch (error) {
      console.error('Failed to switch session:', error)
    }
  }

  const handleDeleteSession = async (session, e) => {
    e.stopPropagation()
    if (!confirm(`Delete this chat session?`)) return

    try {
      await api.delete(`/session/${session.session_id}`)
      loadSessions()
      if (session.session_id === sessionId) {
        handleNewSession()
      }
    } catch (error) {
      console.error('Failed to delete session:', error)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    
    if (draggedDocument) {
      setAttachedDocument(draggedDocument)
      onDragEnd()
    }
  }

  const removeAttachedDocument = () => {
    setAttachedDocument(null)
  }

  const formatTime = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-chatgpt-200 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-5 h-5 text-chatgpt-700" />
          <h2 className="text-lg font-semibold text-chatgpt-900">Chat</h2>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowSessions(!showSessions)}
            className="p-2 hover:bg-chatgpt-100 rounded-lg transition-colors"
            title="View sessions"
          >
            <MessageSquare className="w-4 h-4 text-chatgpt-600" />
          </button>
          <button
            onClick={handleNewSession}
            className="p-2 hover:bg-chatgpt-100 rounded-lg transition-colors"
            title="New chat"
          >
            <Plus className="w-4 h-4 text-chatgpt-600" />
          </button>
        </div>
      </div>

      {/* Sessions Panel */}
      {showSessions && (
        <div className="p-4 border-b border-chatgpt-200 bg-chatgpt-50 max-h-48 overflow-y-auto">
          <h3 className="text-sm font-semibold text-chatgpt-700 mb-2">Recent Sessions</h3>
          {sessions.length === 0 ? (
            <p className="text-xs text-chatgpt-500">No sessions yet</p>
          ) : (
            <div className="space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  onClick={() => handleSwitchSession(session)}
                  className="group flex items-center justify-between p-2 bg-white hover:bg-chatgpt-100 rounded-lg cursor-pointer transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-chatgpt-900 truncate">{session.name}</p>
                    <p className="text-xs text-chatgpt-500">
                      {session.message_count} message{session.message_count !== 1 ? 's' : ''}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(session, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                  >
                    <Trash2 className="w-3 h-3 text-red-500" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Messages Area */}
      <div
        ref={chatAreaRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          flex-1 overflow-y-auto p-4 space-y-4
          ${isDragOver ? 'bg-blue-50 border-2 border-blue-300 border-dashed' : ''}
        `}
      >
        {isDragOver && (
          <div className="text-center py-8">
            <FileText className="w-12 h-12 mx-auto mb-2 text-blue-500" />
            <p className="text-sm text-blue-700 font-medium">
              Drop document to attach
            </p>
            <p className="text-xs text-blue-600 mt-1">
              Chat will use only this document
            </p>
          </div>
        )}

        {!indexStatus && messages.length === 0 && (
          <div className="text-center py-8">
            <MessageSquare className="w-12 h-12 mx-auto mb-2 text-chatgpt-300" />
            <p className="text-sm text-chatgpt-600 font-medium">
              Index not built
            </p>
            <p className="text-xs text-chatgpt-500 mt-1">
              Build the index to start chatting
            </p>
          </div>
        )}

        {messages.length === 0 && indexStatus && !isDragOver && (
          <div className="text-center py-8">
            <MessageSquare className="w-12 h-12 mx-auto mb-2 text-chatgpt-300" />
            <p className="text-sm text-chatgpt-600 font-medium">
              Start a conversation
            </p>
            <p className="text-xs text-chatgpt-500 mt-1">
              Ask anything about your documents
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`
              flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}
            `}
          >
            <div
              className={`
                max-w-[85%] rounded-lg p-3
                ${message.role === 'user'
                  ? 'bg-chatgpt-900 text-white'
                  : 'bg-white border border-chatgpt-200 text-chatgpt-900'
                }
              `}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              
              {/* Assistant message metadata */}
              {message.role === 'assistant' && (
                <div className="mt-2 pt-2 border-t border-chatgpt-200">
                  <div className="flex items-center justify-between text-xs text-chatgpt-500">
                    <span>
                      Confidence: {message.confidence ? (message.confidence * 100).toFixed(0) : 0}%
                    </span>
                    <span>{formatTime(message.timestamp)}</span>
                  </div>
                  
                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-chatgpt-600 font-medium mb-1">Sources:</p>
                      {message.sources.slice(0, 2).map((source, idx) => (
                        <div key={idx} className="text-xs text-chatgpt-500 flex items-start space-x-1 mt-1">
                          <Link2 className="w-3 h-3 flex-shrink-0 mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <span className="truncate">{source.source}</span>
                            {source.similarity_percent !== undefined && (
                              <span className="ml-2 text-chatgpt-400">
                                ({source.similarity_percent}% match)
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* User message time */}
              {message.role === 'user' && (
                <p className="text-xs text-chatgpt-300 mt-1 text-right">
                  {formatTime(message.timestamp)}
                </p>
              )}
            </div>
          </div>
        ))}

        {sending && (
          <div className="flex justify-start">
            <div className="bg-white border border-chatgpt-200 rounded-lg p-3">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-chatgpt-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-chatgpt-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-chatgpt-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Attached Document */}
      {attachedDocument && (
        <div className="px-4 py-2 border-t border-chatgpt-200 bg-blue-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-blue-800 font-medium">
                Using: {attachedDocument.name}
              </span>
            </div>
            <button
              onClick={removeAttachedDocument}
              className="p-1 hover:bg-blue-100 rounded transition-colors"
            >
              <X className="w-4 h-4 text-blue-600" />
            </button>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-t border-chatgpt-200 bg-white">
        <div className="flex space-x-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              !indexStatus 
                ? "Build index first..." 
                : attachedDocument 
                ? `Ask about ${attachedDocument.name}...`
                : "Ask about your documents..."
            }
            disabled={!indexStatus || sending}
            className="flex-1 resize-none rounded-lg border border-chatgpt-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-chatgpt-500 focus:border-transparent disabled:bg-chatgpt-100 disabled:cursor-not-allowed"
            rows={3}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || sending || !indexStatus}
            className="px-4 bg-chatgpt-900 text-white rounded-lg hover:bg-chatgpt-800 disabled:bg-chatgpt-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-chatgpt-500 mt-2">
          {attachedDocument 
            ? `🔒 Chat will use only "${attachedDocument.name}"`
            : "💡 Drag a document here to ask about it specifically"
          }
        </p>
      </div>
    </div>
  )
}

export default ChatSidebar


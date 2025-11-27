import { useState, useEffect } from 'react'
import { FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Document, Page, pdfjs } from 'react-pdf'
import api from '../services/api'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`

function DocumentViewer({ document }) {
  const [content, setContent] = useState(null)
  const [loading, setLoading] = useState(false)
  const [numPages, setNumPages] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)

  useEffect(() => {
    if (document) {
      loadDocument()
    } else {
      setContent(null)
    }
  }, [document])

  const loadDocument = async () => {
    setLoading(true)
    try {
      const response = await api.get('/file-content', {
        params: { path: document.path }
      })
      setContent(response.data)
      setPageNumber(1)
    } catch (error) {
      console.error('Failed to load document:', error)
      setContent({ type: 'error', message: 'Failed to load document' })
    } finally {
      setLoading(false)
    }
  }

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages)
  }

  const changePage = (offset) => {
    setPageNumber(prevPageNumber => prevPageNumber + offset)
  }

  const previousPage = () => {
    changePage(-1)
  }

  const nextPage = () => {
    changePage(1)
  }

  if (!document) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <FileText className="w-16 h-16 mx-auto mb-4 text-chatgpt-300" />
          <p className="text-lg text-chatgpt-600 font-medium">No document selected</p>
          <p className="text-sm text-chatgpt-500 mt-2">
            Select a document from the sidebar to view
          </p>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-chatgpt-900 mx-auto mb-4"></div>
          <p className="text-chatgpt-600">Loading document...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Document Header */}
      <div className="p-4 border-b border-chatgpt-200 bg-white">
        <h3 className="text-lg font-semibold text-chatgpt-900">{document.name}</h3>
        <p className="text-sm text-chatgpt-500 mt-1">
          {document.type.toUpperCase()} • {(document.size / 1024).toFixed(1)} KB
        </p>
      </div>

      {/* Document Content */}
      <div className="flex-1 overflow-auto p-6 bg-chatgpt-50">
        {content?.type === 'markdown' && (
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-sm p-8">
            <ReactMarkdown
              className="prose prose-chatgpt max-w-none"
              components={{
                h1: ({node, ...props}) => <h1 className="text-3xl font-bold text-chatgpt-900 mt-6 mb-4" {...props} />,
                h2: ({node, ...props}) => <h2 className="text-2xl font-bold text-chatgpt-900 mt-5 mb-3" {...props} />,
                h3: ({node, ...props}) => <h3 className="text-xl font-semibold text-chatgpt-900 mt-4 mb-2" {...props} />,
                p: ({node, ...props}) => <p className="text-chatgpt-700 leading-relaxed mb-4" {...props} />,
                code: ({node, inline, ...props}) => 
                  inline 
                    ? <code className="bg-chatgpt-100 text-chatgpt-800 px-1 py-0.5 rounded text-sm" {...props} />
                    : <code className="block bg-chatgpt-900 text-white p-4 rounded-lg overflow-x-auto mb-4" {...props} />,
                ul: ({node, ...props}) => <ul className="list-disc list-inside text-chatgpt-700 mb-4 space-y-2" {...props} />,
                ol: ({node, ...props}) => <ol className="list-decimal list-inside text-chatgpt-700 mb-4 space-y-2" {...props} />,
                a: ({node, ...props}) => <a className="text-blue-600 hover:text-blue-800 underline" {...props} />,
              }}
            >
              {content.content}
            </ReactMarkdown>
          </div>
        )}

        {content?.type === 'pdf' && (
          <div className="flex flex-col items-center">
            <Document
              file={`/api/file?path=${encodeURIComponent(content.path || document.path)}`}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={(error) => {
                console.error('PDF load error:', error)
                setContent({ type: 'error', message: 'Failed to load PDF. Please check the console for details.' })
              }}
              loading={
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-chatgpt-900 mx-auto mb-4"></div>
                  <p className="text-chatgpt-600">Loading PDF...</p>
                </div>
              }
              error={
                <div className="text-center py-8">
                  <FileText className="w-12 h-12 mx-auto mb-4 text-red-500" />
                  <p className="text-red-600 font-medium">Failed to load PDF</p>
                  <p className="text-sm text-chatgpt-500 mt-2">Please check that the file exists and the backend is running</p>
                </div>
              }
              className="shadow-lg"
            >
              <Page 
                pageNumber={pageNumber} 
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="max-w-full"
                width={Math.min(window.innerWidth * 0.6, 800)}
              />
            </Document>
            
            {/* PDF Navigation */}
            <div className="mt-4 flex items-center space-x-4 bg-white px-4 py-2 rounded-lg shadow">
              <button
                onClick={previousPage}
                disabled={pageNumber <= 1}
                className="px-3 py-1 bg-chatgpt-900 text-white rounded hover:bg-chatgpt-800 disabled:bg-chatgpt-300 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-chatgpt-700">
                Page {pageNumber} of {numPages}
              </span>
              <button
                onClick={nextPage}
                disabled={pageNumber >= numPages}
                className="px-3 py-1 bg-chatgpt-900 text-white rounded hover:bg-chatgpt-800 disabled:bg-chatgpt-300 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {content?.type === 'docx' && (
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-sm p-8">
            <p className="text-center text-chatgpt-600">
              DOCX preview coming soon...
            </p>
            <p className="text-center text-sm text-chatgpt-500 mt-2">
              Download the file to view
            </p>
          </div>
        )}

        {content?.type === 'error' && (
          <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-sm p-8">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 text-red-500" />
              <p className="text-lg text-red-600 font-medium">{content.message || 'Error loading document'}</p>
              <p className="text-sm text-chatgpt-500 mt-2">
                Please check the browser console for more details
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DocumentViewer


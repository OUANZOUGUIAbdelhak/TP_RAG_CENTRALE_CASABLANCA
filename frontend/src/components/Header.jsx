import { RefreshCw, Database, CheckCircle, XCircle, Loader2, Trash2 } from 'lucide-react'

function Header({ indexStatus, onBuildIndex, onRefresh, buildingIndex, buildProgress, onDeleteVectorstore }) {
  return (
    <div className="h-14 bg-white border-b border-chatgpt-200 flex items-center justify-between px-6 relative">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-semibold text-chatgpt-900">
          RAG System
        </h1>
        <span className="text-sm text-chatgpt-500">
          École Centrale Casablanca
        </span>
      </div>

      <div className="flex items-center space-x-4">
        {/* Index Status */}
        <div className="flex items-center space-x-2">
          {indexStatus ? (
            <>
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span className="text-sm text-chatgpt-700">Index Ready</span>
            </>
          ) : (
            <>
              <XCircle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-chatgpt-700">Index Not Built</span>
            </>
          )}
        </div>

        {/* Build Index Button */}
        <button
          onClick={onBuildIndex}
          disabled={buildingIndex}
          className="flex items-center space-x-2 px-4 py-2 bg-chatgpt-900 text-white rounded-lg hover:bg-chatgpt-800 transition-colors disabled:bg-chatgpt-400 disabled:cursor-not-allowed"
        >
          {buildingIndex ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">Building...</span>
            </>
          ) : (
            <>
              <Database className="w-4 h-4" />
              <span className="text-sm font-medium">Build Index</span>
            </>
          )}
        </button>
        
        {/* Build Progress */}
        {buildingIndex && buildProgress.message && (
          <div className="absolute top-16 right-6 bg-white border border-chatgpt-200 rounded-lg shadow-lg p-4 min-w-[300px] z-50">
            <div className="flex items-center space-x-2 mb-2">
              <Loader2 className="w-4 h-4 animate-spin text-chatgpt-600" />
              <span className="text-sm font-medium text-chatgpt-900">
                {buildProgress.message}
              </span>
            </div>
            <div className="w-full bg-chatgpt-100 rounded-full h-2">
              <div 
                className="bg-chatgpt-900 h-2 rounded-full transition-all duration-300"
                style={{ width: `${buildProgress.progress}%` }}
              ></div>
            </div>
            <p className="text-xs text-chatgpt-500 mt-2">
              {buildProgress.progress}% complete
            </p>
          </div>
        )}

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          className="p-2 hover:bg-chatgpt-100 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5 text-chatgpt-600" />
        </button>

        {/* Delete Vectorstore Button */}
        {indexStatus && (
          <button
            onClick={onDeleteVectorstore}
            className="p-2 hover:bg-red-100 rounded-lg transition-colors"
            title="Delete vectorstore and start fresh"
          >
            <Trash2 className="w-5 h-5 text-red-600" />
          </button>
        )}
      </div>
    </div>
  )
}

export default Header


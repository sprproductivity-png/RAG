import React, { useState, useEffect } from 'react';
import { Files, Trash2, RefreshCw, FileText, Search, ExternalLink, AlertCircle, Eye, X } from 'lucide-react';
import { fetchDocuments, deleteDocumentApi } from '../api';

export default function DocumentManager({ collections, selectedCollection, onSelectCollection, showToast }) {
  const [activeCollection, setActiveCollection] = useState(
    selectedCollection || (collections.length > 0 ? collections[0].collection_name : 'documents')
  );
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [previewDoc, setPreviewDoc] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  // Sync active collection if collections list changes
  useEffect(() => {
    if (!activeCollection && collections.length > 0) {
      setActiveCollection(collections[0].collection_name);
    }
  }, [collections, activeCollection]);

  const loadDocuments = async (collName) => {
    const target = collName || activeCollection;
    if (!target) return;
    setLoading(true);
    try {
      const data = await fetchDocuments(target);
      setDocuments(data.documents || []);
    } catch (err) {
      showToast(err.message, true);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeCollection) {
      loadDocuments(activeCollection);
    }
  }, [activeCollection]);

  const handleDelete = async (doc) => {
    if (!window.confirm(`Delete document "${doc.title}" (ID: ${doc.doc_id}) and all its ${doc.chunks_count} chunk(s) from collection "${activeCollection}"?`)) {
      return;
    }

    setDeletingId(doc.doc_id);
    try {
      const res = await deleteDocumentApi(activeCollection, doc.doc_id);
      showToast(res.message || `Document deleted (${res.deleted_chunks} chunks removed)`);
      // Update local state without waiting for refetch
      setDocuments((prev) => prev.filter((d) => d.doc_id !== doc.doc_id));
      if (previewDoc?.doc_id === doc.doc_id) {
        setPreviewDoc(null);
      }
    } catch (err) {
      showToast(err.message, true);
    } finally {
      setDeletingId(null);
    }
  };

  const filteredDocs = documents.filter((doc) => {
    const q = searchTerm.toLowerCase();
    return (
      (doc.title && doc.title.toLowerCase().includes(q)) ||
      (doc.doc_id && doc.doc_id.toLowerCase().includes(q)) ||
      (doc.filename && doc.filename.toLowerCase().includes(q))
    );
  });

  const getBadgeStyle = (fileType) => {
    switch ((fileType || '').toLowerCase()) {
      case '.pdf':
        return 'bg-rose-950/60 text-rose-300 border-rose-800/40';
      case '.json':
        return 'bg-amber-950/60 text-amber-300 border-amber-800/40';
      case '.csv':
      case '.tsv':
        return 'bg-emerald-950/60 text-emerald-300 border-emerald-800/40';
      case '.md':
      case '.markdown':
        return 'bg-sky-950/60 text-sky-300 border-sky-800/40';
      default:
        return 'bg-indigo-950/60 text-indigo-300 border-indigo-800/40';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Controls Card */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="font-bold text-base text-slate-100 flex items-center">
              <Files className="w-5 h-5 mr-2 text-indigo-400" />
              Document Manager
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect, preview, and selectively delete indexed documents and their vector chunks without dropping the collection.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => loadDocuments(activeCollection)}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition flex items-center space-x-1.5 disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Filter bar: Collection selector & Search input */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-slate-800">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Target Collection</label>
            <select
              value={activeCollection}
              onChange={(e) => {
                setActiveCollection(e.target.value);
                if (onSelectCollection) onSelectCollection(e.target.value);
              }}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-medium"
            >
              {collections.length === 0 ? (
                <option value="documents">documents (default)</option>
              ) : (
                collections.map((c) => (
                  <option key={c.collection_name} value={c.collection_name}>
                    {c.collection_name} ({c.points_count} vectors)
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="sm:col-span-2">
            <label className="block text-xs font-medium text-slate-400 mb-1">Filter Documents</label>
            <div className="relative">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by title, filename, or document ID..."
                className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
              />
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              {searchTerm && (
                <button
                  onClick={() => setSearchTerm('')}
                  className="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300 text-xs"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Documents Table */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div>
            Showing <span className="font-semibold text-slate-200">{filteredDocs.length}</span> of{' '}
            <span className="font-semibold text-slate-200">{documents.length}</span> documents in{' '}
            <span className="font-mono text-indigo-400">'{activeCollection}'</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="text-[11px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Document Title & ID</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4">Chunks</th>
                <th className="py-3 px-4">Ingested At</th>
                <th className="py-3 px-4">Preview Snippet</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="6" className="text-center py-12">
                    <div className="inline-flex items-center space-x-2 text-slate-400">
                      <div className="spinner"></div>
                      <span>Loading documents...</span>
                    </div>
                  </td>
                </tr>
              ) : filteredDocs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="text-center py-12 text-slate-400">
                    {searchTerm ? (
                      <div>No documents match search "{searchTerm}".</div>
                    ) : (
                      <div className="space-y-1">
                        <p className="font-medium text-slate-300">No documents indexed in this collection yet.</p>
                        <p className="text-slate-500">Go to the "Ingestion & Chunking Studio" tab to upload files or paste text.</p>
                      </div>
                    )}
                  </td>
                </tr>
              ) : (
                filteredDocs.map((doc) => (
                  <tr key={doc.doc_id} className="border-b border-slate-800/80 hover:bg-slate-800/30 transition">
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-200 flex items-center space-x-1.5">
                        <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                        <span>{doc.title || doc.filename || 'Untitled'}</span>
                      </div>
                      <div className="text-[11px] font-mono text-slate-500 mt-0.5">
                        ID: {doc.doc_id}
                        {doc.filename && doc.filename !== doc.title && (
                          <span className="ml-2 text-slate-400">({doc.filename})</span>
                        )}
                      </div>
                    </td>

                    <td className="py-3 px-4">
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getBadgeStyle(doc.file_type)}`}>
                        {doc.file_type ? doc.file_type.replace('.', '') : 'TEXT'}
                      </span>
                    </td>

                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-slate-800 text-indigo-300 border border-slate-700">
                        {doc.chunks_count} {doc.chunks_count === 1 ? 'chunk' : 'chunks'}
                      </span>
                    </td>

                    <td className="py-3 px-4 text-slate-400 whitespace-nowrap">
                      {doc.ingested_at ? new Date(doc.ingested_at).toLocaleString() : '—'}
                    </td>

                    <td className="py-3 px-4 max-w-xs truncate text-slate-400">
                      {doc.sample_preview || 'No preview available'}
                    </td>

                    <td className="py-3 px-4 text-right whitespace-nowrap">
                      <div className="inline-flex items-center space-x-2">
                        <button
                          onClick={() => setPreviewDoc(doc)}
                          className="text-slate-400 hover:text-slate-200 text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 transition inline-flex items-center space-x-1"
                          title="View Details"
                        >
                          <Eye className="w-3 h-3" />
                          <span>View</span>
                        </button>
                        <button
                          onClick={() => handleDelete(doc)}
                          disabled={deletingId === doc.doc_id}
                          className="text-rose-400 hover:text-rose-300 text-xs px-2.5 py-1 rounded bg-rose-950/40 border border-rose-800/40 hover:bg-rose-900/50 transition inline-flex items-center space-x-1 disabled:opacity-50"
                          title="Delete Document & all chunks"
                        >
                          <Trash2 className="w-3 h-3" />
                          <span>{deletingId === doc.doc_id ? 'Deleting...' : 'Delete'}</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Preview Modal */}
      {previewDoc && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="glass-panel w-full max-w-2xl rounded-2xl border border-slate-800 p-6 space-y-4 max-h-[85vh] flex flex-col">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                <div>
                  <h4 className="font-bold text-slate-100 text-sm">{previewDoc.title || 'Document Preview'}</h4>
                  <div className="text-[11px] font-mono text-slate-500">ID: {previewDoc.doc_id}</div>
                </div>
              </div>
              <button
                onClick={() => setPreviewDoc(null)}
                className="text-slate-400 hover:text-slate-200 p-1 rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <div>
                <span className="text-slate-500 block">Collection</span>
                <span className="font-medium text-slate-200">{activeCollection}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Chunks</span>
                <span className="font-medium text-indigo-400">{previewDoc.chunks_count}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Format</span>
                <span className="font-medium text-slate-200 uppercase">{previewDoc.file_type || 'TEXT'}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Filename</span>
                <span className="font-medium text-slate-200 truncate block">{previewDoc.filename || '—'}</span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2">
              <label className="text-xs font-semibold text-slate-400">Content Snippet</label>
              <div className="bg-slate-900 p-3.5 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap">
                {previewDoc.sample_preview || 'No preview text stored for this document.'}
              </div>

              {previewDoc.metadata && Object.keys(previewDoc.metadata).length > 0 && (
                <div className="pt-2">
                  <label className="text-xs font-semibold text-slate-400">Metadata</label>
                  <pre className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-400 overflow-x-auto mt-1">
                    {JSON.stringify(previewDoc.metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-800">
              <button
                onClick={() => handleDelete(previewDoc)}
                className="px-3.5 py-1.5 rounded-lg bg-rose-950/60 border border-rose-800/40 text-rose-300 hover:bg-rose-900/60 text-xs font-medium flex items-center space-x-1.5 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Delete Document</span>
              </button>

              <button
                onClick={() => setPreviewDoc(null)}
                className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

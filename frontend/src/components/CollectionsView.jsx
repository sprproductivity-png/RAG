import React from 'react';
import { Database, Trash2, RefreshCw } from 'lucide-react';
import { deleteCollectionApi } from '../api';

export default function CollectionsView({ collections, onRefresh, onViewDocuments, showToast }) {
  const handleDelete = async (name) => {
    if (!window.confirm(`Are you sure you want to delete collection '${name}' and its BM25 index?`)) {
      return;
    }

    try {
      await deleteCollectionApi(name);
      showToast(`Collection '${name}' deleted successfully.`);
      if (onRefresh) onRefresh();
    } catch (err) {
      showToast(err.message, true);
    }
  };

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-bold text-base text-slate-100 flex items-center">
            <Database className="w-5 h-5 mr-2 text-indigo-400" />
            Vector Collections & BM25 Indices
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time status of vector indices stored in Qdrant and synchronized BM25Plus files.
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition flex items-center space-x-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="text-[11px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Collection Name</th>
              <th className="py-3 px-4">Qdrant Points</th>
              <th className="py-3 px-4">Vector Dim</th>
              <th className="py-3 px-4">Metric</th>
              <th className="py-3 px-4">BM25 Docs</th>
              <th className="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {collections.length === 0 ? (
              <tr>
                <td colSpan="6" className="text-center py-8 text-slate-400">
                  No collections found. Ingest a document in the Ingestion Studio tab to create one!
                </td>
              </tr>
            ) : (
              collections.map((c) => (
                <tr key={c.collection_name} className="border-b border-slate-800/80 hover:bg-slate-800/30">
                  <td className="py-3 px-4 font-medium text-indigo-300">{c.collection_name}</td>
                  <td className="py-3 px-4 text-slate-300">{c.points_count}</td>
                  <td className="py-3 px-4 text-slate-300">{c.vector_size}</td>
                  <td className="py-3 px-4 text-slate-300">{c.distance}</td>
                  <td className="py-3 px-4 text-slate-300">{c.bm25_docs_count}</td>
                  <td className="py-3 px-4 text-right">
                    <div className="inline-flex items-center space-x-2">
                      {onViewDocuments && (
                        <button
                          onClick={() => onViewDocuments(c.collection_name)}
                          className="text-indigo-400 hover:text-indigo-300 text-xs px-2.5 py-1 rounded bg-indigo-950/40 border border-indigo-800/40 hover:bg-indigo-900/50 transition inline-flex items-center space-x-1"
                          title="View and manage individual documents"
                        >
                          <span>Documents</span>
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(c.collection_name)}
                        className="text-rose-400 hover:text-rose-300 text-xs px-2.5 py-1 rounded bg-rose-950/40 border border-rose-800/40 hover:bg-rose-900/50 transition inline-flex items-center space-x-1"
                        title="Delete entire collection"
                      >
                        <Trash2 className="w-3 h-3" />
                        <span>Delete</span>
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
  );
}


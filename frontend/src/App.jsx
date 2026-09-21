import React, { useState, useEffect } from 'react';
import { Search, UploadCloud, Layers, Database } from 'lucide-react';
import Navbar from './components/Navbar';
import SearchStudio from './components/SearchStudio';
import IngestStudio from './components/IngestStudio';
import RerankPlayground from './components/RerankPlayground';
import CollectionsView from './components/CollectionsView';
import { fetchHealth, fetchCollections } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('search');
  const [health, setHealth] = useState(null);
  const [collections, setCollections] = useState([]);
  const [toast, setToast] = useState(null);

  const showToast = (message, isError = false) => {
    setToast({ message, isError });
    setTimeout(() => {
      setToast(null);
    }, 4000);
  };

  const loadData = async () => {
    try {
      const [healthData, collsData] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchCollections().catch(() => ({ collections: [] })),
      ]);
      if (healthData) setHealth(healthData);
      if (collsData?.collections) setCollections(collsData.collections);
    } catch (err) {
      console.error('Initial load error:', err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => {});
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased">
      {/* Navbar Header */}
      <Navbar health={health} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Tab Navigation */}
        <div className="flex items-center space-x-2 mb-6 border-b border-slate-800/80 pb-3 overflow-x-auto">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition flex items-center space-x-2 shrink-0 ${
              activeTab === 'search'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Search className="w-4 h-4" />
            <span>Hybrid Search & Analysis</span>
          </button>

          <button
            onClick={() => setActiveTab('ingest')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition flex items-center space-x-2 shrink-0 ${
              activeTab === 'ingest'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <UploadCloud className="w-4 h-4" />
            <span>Ingestion & Chunking Studio</span>
          </button>

          <button
            onClick={() => setActiveTab('rerank')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition flex items-center space-x-2 shrink-0 ${
              activeTab === 'rerank'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Cross-Encoder Playground</span>
          </button>

          <button
            onClick={() => {
              setActiveTab('collections');
              fetchCollections().then((d) => setCollections(d.collections || []));
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition flex items-center space-x-2 shrink-0 ${
              activeTab === 'collections'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-4 h-4" />
            <span>Collections & Stats</span>
          </button>
        </div>

        {/* Tab Views */}
        {activeTab === 'search' && (
          <SearchStudio collections={collections} showToast={showToast} />
        )}

        {activeTab === 'ingest' && (
          <IngestStudio
            onIngestSuccess={() => {
              fetchCollections().then((d) => setCollections(d.collections || []));
            }}
            showToast={showToast}
          />
        )}

        {activeTab === 'rerank' && (
          <RerankPlayground showToast={showToast} />
        )}

        {activeTab === 'collections' && (
          <CollectionsView
            collections={collections}
            onRefresh={() => fetchCollections().then((d) => setCollections(d.collections || []))}
            showToast={showToast}
          />
        )}
      </main>

      {/* Floating Toast Notification */}
      {toast && (
        <div
          className={`fixed bottom-6 right-6 px-4 py-3 rounded-xl text-sm font-medium shadow-2xl z-50 transition-all duration-300 ${
            toast.isError ? 'bg-rose-600 text-white' : 'bg-emerald-600 text-white'
          }`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}


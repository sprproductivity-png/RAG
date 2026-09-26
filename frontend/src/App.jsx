import React, { useState, useEffect } from 'react';
import { Search, UploadCloud, Layers, Database, Files, Loader2 } from 'lucide-react';
import Navbar from './components/Navbar';
import SearchStudio from './components/SearchStudio';
import IngestStudio from './components/IngestStudio';
import RerankPlayground from './components/RerankPlayground';
import CollectionsView from './components/CollectionsView';
import DocumentManager from './components/DocumentManager';
import { fetchHealth, fetchCollections, fetchTasks, fetchTaskStatus } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('search');
  const [selectedCollection, setSelectedCollection] = useState('documents');
  const [health, setHealth] = useState(null);
  const [collections, setCollections] = useState([]);
  const [toast, setToast] = useState(null);

  // Background Ingestion Task Management (Global across all tabs)
  const [activeTask, setActiveTask] = useState(null);
  const [taskList, setTaskList] = useState([]);
  const [sampleChunks, setSampleChunks] = useState(null);

  const showToast = (message, isError = false) => {
    setToast({ message, isError });
    setTimeout(() => {
      setToast(null);
    }, 4000);
  };

  const loadTasks = async () => {
    try {
      const res = await fetchTasks();
      const list = res.tasks || [];
      setTaskList(list);

      // Check if there is an active running task
      const savedTaskId = localStorage.getItem('rag_active_task_id');
      if (savedTaskId) {
        const found = list.find((t) => t.task_id === savedTaskId);
        if (found) {
          setActiveTask(found);
          if (found.sample_chunks) setSampleChunks(found.sample_chunks);
          return;
        }
      }

      // If no active task currently selected, pick any task that is running
      if (!activeTask || activeTask.status === 'completed' || activeTask.status === 'failed') {
        const running = list.find((t) => t.status === 'processing' || t.status === 'queued');
        if (running) {
          setActiveTask(running);
          localStorage.setItem('rag_active_task_id', running.task_id);
        }
      }
    } catch (err) {
      console.error('Error fetching tasks history:', err);
    }
  };

  const loadData = async () => {
    try {
      const [healthData, collsData] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchCollections().catch(() => ({ collections: [] })),
      ]);
      if (healthData) setHealth(healthData);
      if (collsData?.collections) setCollections(collsData.collections);
      await loadTasks();
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

  // Global persistent polling loop for active task across all tabs
  useEffect(() => {
    if (!activeTask || activeTask.status === 'completed' || activeTask.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await fetchTaskStatus(activeTask.task_id);
        setActiveTask(updated);

        if (updated.status === 'completed') {
          showToast(`Ingestion completed for "${updated.filename}": created ${updated.chunks_created} chunks in ${updated.duration_ms}ms`);
          setSampleChunks(updated.sample_chunks || []);
          localStorage.removeItem('rag_active_task_id');
          loadTasks();
          fetchCollections().then((d) => setCollections(d.collections || []));
        } else if (updated.status === 'failed') {
          showToast(updated.error || `Task failed during ${updated.stage}`, true);
          localStorage.removeItem('rag_active_task_id');
          loadTasks();
        }
      } catch (err) {
        console.error('Error polling task status:', err);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [activeTask?.task_id, activeTask?.status]);

  const isTaskRunning = activeTask && (activeTask.status === 'queued' || activeTask.status === 'processing');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased">
      {/* Navbar Header */}
      <Navbar health={health} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {/* Global Active Ingestion Banner (Visible on all tabs so user never loses track) */}
        {isTaskRunning && (
          <div className="mb-6 p-4 rounded-2xl bg-indigo-950/60 border border-indigo-500/40 shadow-xl shadow-indigo-950/40 flex flex-col md:flex-row md:items-center justify-between gap-4 animate-fade-in">
            <div className="flex items-center space-x-3.5 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-indigo-600/30 border border-indigo-500/50 flex items-center justify-center text-indigo-400 shrink-0">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-white truncate">
                    Ingesting: {activeTask.filename}
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 uppercase">
                    {activeTask.stage} &bull; {activeTask.progress}%
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 truncate mt-0.5">
                  {activeTask.stage_message}
                </p>
                <div className="w-full bg-slate-900 rounded-full h-1.5 mt-2 border border-slate-800 overflow-hidden max-w-md">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-1.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${Math.max(activeTask.progress, 5)}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2 shrink-0">
              <button
                onClick={() => setActiveTab('ingest')}
                className="px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/20 transition flex items-center space-x-1"
              >
                <span>View in Studio</span>
                <span className="text-xs">&rarr;</span>
              </button>
            </div>
          </div>
        )}

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
            {isTaskRunning && (
              <span className="relative flex h-2 w-2 ml-1">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('documents')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition flex items-center space-x-2 shrink-0 ${
              activeTab === 'documents'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Files className="w-4 h-4" />
            <span>Document Manager</span>
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
            activeTask={activeTask}
            setActiveTask={(task) => {
              setActiveTask(task);
              if (task && (task.status === 'queued' || task.status === 'processing')) {
                localStorage.setItem('rag_active_task_id', task.task_id);
              } else {
                localStorage.removeItem('rag_active_task_id');
              }
            }}
            taskList={taskList}
            loadTasks={loadTasks}
            sampleChunks={sampleChunks}
            setSampleChunks={setSampleChunks}
            onIngestSuccess={() => {
              fetchCollections().then((d) => setCollections(d.collections || []));
            }}
            showToast={showToast}
          />
        )}

        {activeTab === 'documents' && (
          <DocumentManager
            collections={collections}
            selectedCollection={selectedCollection}
            onSelectCollection={(coll) => setSelectedCollection(coll)}
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
            onViewDocuments={(collName) => {
              setSelectedCollection(collName);
              setActiveTab('documents');
            }}
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

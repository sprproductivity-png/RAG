import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, Sparkles, BookOpen, Upload, File, X, AlertCircle } from 'lucide-react';
import { executeIngestApi, uploadFileApi, fetchTaskStatus, fetchTasks } from '../api';
import TaskProgressCard from './TaskProgressCard';
import TaskHistoryTable from './TaskHistoryTable';

const PRESETS = {
  robotics: {
    title: "Acme Robotics Annual Technology Review 2024",
    category: "Autonomous Systems",
    text: `Acme Robotics Inc. - Annual Engineering & Technology Review 2024.

Section 1: Autonomous Navigation and SLAM Algorithms
Our autonomous mobile robots (AMRs) utilize graph-based Simultaneous Localization and Mapping (SLAM). By fusing high-frequency LiDAR scans with wheel odometry and IMU telemetry via an Extended Kalman Filter, the robots maintain sub-centimeter positioning accuracy even in dynamic warehouse environments with moving personnel.

Section 2: Battery Management and Fast Charging Systems
The new Falcon-5 powertrain integrates lithium iron phosphate (LiFePO4) battery packs delivering 18 hours of continuous operation. Our proprietary thermal management system monitors individual cell temperatures across 48 channels, mitigating thermal runaway risks and enabling 80% recharge within 35 minutes via inductive floor pads.

Section 3: Cloud Telemetry and Fleet Coordination
Fleet orchestration is powered by a decentralized consensus algorithm that prevents path deadlocks at narrow aisle intersections. All robot telemetry, including motor currents and wheel slip metrics, is streamed via MQTT over private 5G networks to our centralized analytics dashboard.`
  },
  finance: {
    title: "Apex Global Financial Statement Q3 2024",
    category: "Finance & Earnings",
    text: `Apex Global Financial Holdings - Third Quarter Fiscal Summary 2024.

Part 1: Core Revenue & Cloud Infrastructure Segment
Consolidated net revenues reached $18.4 billion, reflecting a 16% year-over-year expansion. Enterprise cloud services spearheaded quarterly growth, contributing $6.8 billion driven by accelerated enterprise adoption of generative AI workload pipelines.

Part 2: Operating Margins & Cost Optimization
Operating margins widened to 28.5%, supported by automation in customer support and streamlined datacenter cooling architecture. Capital expenditures totaled $3.1 billion, focused on sovereign AI infrastructure in the EU and APAC regions.

Part 3: Shareholder Returns & Capital Allocation
The board authorized an additional $10 billion share repurchase program and declared a quarterly dividend of $0.45 per share. Cash and marketable securities closed the quarter at $34.2 billion.`
  }
};

export default function IngestStudio({
  activeTask,
  setActiveTask,
  taskList = [],
  loadTasks,
  sampleChunks,
  setSampleChunks,
  onIngestSuccess,
  showToast,
  health,
}) {
  const [mode, setMode] = useState('upload'); // 'upload' or 'text'
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('');
  const [text, setText] = useState('');
  const [collection, setCollection] = useState('documents');
  const [strategy, setStrategy] = useState('semantic');
  const [useContextual, setUseContextual] = useState(true);
  const [provider, setProvider] = useState('heuristic');

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (loadTasks) loadTasks();
  }, []);

  const loadPreset = (key) => {
    const preset = PRESETS[key];
    if (!preset) return;
    setTitle(preset.title);
    setCategory(preset.category);
    setText(preset.text);
    setMode('text');
    showToast(`Loaded preset: ${preset.title}`);
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    if (!title) {
      const base = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
      setTitle(base.replace(/[-_]/g, ' '));
    }
  };

  const handleIngest = async () => {
    if (mode === 'upload') {
      if (!selectedFile) {
        showToast('Please select a file to upload (.pdf, .txt, .md, .csv, .json)', true);
        return;
      }

      setLoading(true);
      try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('collection_name', collection);
        if (title.trim()) formData.append('title', title.trim());
        formData.append('chunking_strategy', strategy);
        formData.append('use_contextual_retrieval', useContextual ? 'true' : 'false');
        if (provider) formData.append('context_provider', provider);
        formData.append('async_processing', 'true');

        // Step 1: Upload & Cache instantly
        const taskData = await uploadFileApi(formData);
        showToast(`File cached on server (${formatFileSize(selectedFile.size)}). Background ingestion queued!`);
        setActiveTask(taskData);
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        loadTasks();
      } catch (err) {
        showToast(err.message, true);
      } finally {
        setLoading(false);
      }
    } else {
      // Raw text mode (synchronous pipeline)
      if (!text.trim()) {
        showToast('Please provide document text to ingest', true);
        return;
      }

      setLoading(true);
      try {
        const data = await executeIngestApi({
          collection_name: collection,
          chunking_strategy: strategy,
          use_contextual_retrieval: useContextual,
          context_provider: provider,
          documents: [
            {
              title: title || 'Untitled Document',
              text: text,
              metadata: {
                title: title || 'Untitled',
                category: category || 'General',
                ingested_at: new Date().toISOString(),
              },
            },
          ],
        });

        showToast(`Ingested ${data.chunks_created} chunks into '${data.collection_name}' in ${data.duration_ms}ms`);
        setSampleChunks(data.sample_chunks || []);
        if (onIngestSuccess) onIngestSuccess();
      } catch (err) {
        showToast(err.message, true);
      } finally {
        setLoading(false);
      }
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Real-time Ingestion Progress Card (Step 2 state) */}
      {activeTask && (
        <TaskProgressCard
          task={activeTask}
          onClear={() => setActiveTask(null)}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Form Column */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
            {/* Header & Mode Switcher */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-bold text-base text-slate-100 flex items-center">
                  <UploadCloud className="w-5 h-5 mr-2 text-indigo-400" />
                  Document Ingestion & Chunking
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  2-step pipeline: raw files are immediately cached on disk, then processed through background task queue.
                </p>
              </div>

              {/* Mode Toggle: File Upload vs Raw Text */}
              <div className="inline-flex rounded-lg bg-slate-900 p-1 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setMode('upload')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition flex items-center space-x-1.5 ${
                    mode === 'upload'
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload File (Async Queue)</span>
                </button>
                <button
                  type="button"
                  onClick={() => setMode('text')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition flex items-center space-x-1.5 ${
                    mode === 'text'
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Paste Text</span>
                </button>
              </div>
            </div>

            {/* Content Input Area: File Dropzone or Text Editor */}
            {mode === 'upload' ? (
              <div className="space-y-4">
                {/* Drag-and-drop Dropzone */}
                <div
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleFileDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition flex flex-col items-center justify-center space-y-3 ${
                    isDragging
                      ? 'border-indigo-500 bg-indigo-950/20'
                      : selectedFile
                      ? 'border-indigo-600/50 bg-slate-900/60'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-900/30'
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.md,.markdown,.csv,.tsv,.json"
                    onChange={(e) => e.target.files?.[0] && handleFileSelected(e.target.files[0])}
                    className="hidden"
                  />

                  {selectedFile ? (
                    <div className="flex items-center space-x-3 text-left p-3 rounded-xl bg-slate-900 border border-slate-700 max-w-md w-full">
                      <File className="w-8 h-8 text-indigo-400 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-slate-200 truncate">{selectedFile.name}</div>
                        <div className="text-[11px] text-slate-400">
                          {formatFileSize(selectedFile.size)} • {selectedFile.name.split('.').pop()?.toUpperCase()}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFile(null);
                          if (fileInputRef.current) fileInputRef.current.value = '';
                        }}
                        className="p-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="w-12 h-12 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 flex items-center justify-center text-indigo-400">
                        <UploadCloud className="w-6 h-6" />
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-slate-200">
                          Drag and drop large file here, or <span className="text-indigo-400 underline">browse files</span>
                        </p>
                        <p className="text-[11px] text-slate-400">
                          Supports: PDF, Markdown (.md), Plain Text (.txt), CSV, JSON
                        </p>
                      </div>
                    </>
                  )}
                </div>

                {/* Optional custom title */}
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">
                    Document Title <span className="text-slate-500 font-normal">(Optional, inferred from filename if empty)</span>
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. Acme Robotics Annual Review 2024"
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Presets row */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400">Load sample document:</span>
                  <div className="flex items-center space-x-2">
                    <button
                      type="button"
                      onClick={() => loadPreset('robotics')}
                      className="px-2.5 py-1 rounded text-xs bg-indigo-950/80 text-indigo-300 border border-indigo-800/40 hover:bg-indigo-900/60 transition"
                    >
                      Acme Robotics
                    </button>
                    <button
                      type="button"
                      onClick={() => loadPreset('finance')}
                      className="px-2.5 py-1 rounded text-xs bg-purple-950/80 text-purple-300 border border-purple-800/40 hover:bg-purple-900/60 transition"
                    >
                      Apex Finance
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Document Title</label>
                    <input
                      type="text"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder="e.g. Acme Robotics Annual Review"
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Category / Topic</label>
                    <input
                      type="text"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      placeholder="e.g. Robotics, Engineering"
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Document Content</label>
                  <textarea
                    rows="8"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste full document text here. The engine will detect semantic breakpoints and prepend Anthropic situational context summaries..."
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono leading-relaxed"
                  />
                </div>
              </div>
            )}

            {/* Pipeline Configuration */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-slate-800">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Target Collection</label>
                <input
                  type="text"
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Chunking Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100"
                >
                  <option value="semantic">Semantic (Embedding Breakpoints)</option>
                  <option value="sliding_window">Sliding Window (Fixed)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Contextual Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100"
                >
                  <option value="heuristic">Zero-Key Offline (Heuristic)</option>
                  <option value="anthropic">Anthropic Claude API</option>
                  <option value="openai">OpenAI / GPT</option>
                  <option value="ollama">Ollama (Local LLM)</option>
                </select>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useContextual}
                  onChange={(e) => setUseContextual(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0"
                />
                <span className="text-xs text-slate-300 font-medium">
                  Prepend Anthropic Contextual Retrieval Summaries before embedding
                </span>
              </label>

              <button
                onClick={handleIngest}
                disabled={loading}
                className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs shadow-lg shadow-indigo-500/25 transition flex items-center disabled:opacity-50"
              >
                {loading ? <div className="spinner mr-2"></div> : <UploadCloud className="w-4 h-4 mr-1.5" />}
                <span>{mode === 'upload' ? 'Upload & Queue Ingestion' : 'Ingest Document'}</span>
              </button>
            </div>
          </div>

          {/* Queue History Table */}
          <TaskHistoryTable
            tasks={taskList}
            onRefresh={loadTasks}
            onSelectTask={(t) => {
              setActiveTask(t);
              if (t.sample_chunks) setSampleChunks(t.sample_chunks);
            }}
            showToast={showToast}
          />
        </div>

        {/* Preview / Info Column */}
        <div className="lg:col-span-1 space-y-4">
          {sampleChunks && sampleChunks.length > 0 && (
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3 animate-fade-in">
              <h4 className="font-bold text-xs uppercase tracking-wider text-indigo-400 flex items-center justify-between">
                <span>Ingested Chunks Preview</span>
                <span className="text-[10px] text-slate-400">{sampleChunks.length} Sample Chunks</span>
              </h4>
              <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
                {sampleChunks.map((c) => (
                  <div key={c.chunk_id} className="glass-panel p-3.5 rounded-xl border border-slate-800 text-xs space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-indigo-400">Chunk #{c.chunk_index}</span>
                      <span className="text-slate-400 font-mono">{c.chunk_id?.slice(0, 8)}...</span>
                    </div>
                    {c.context_summary && (
                      <div className="badge-context p-2.5 rounded-lg text-purple-200">
                        <div className="font-semibold flex items-center mb-0.5">
                          <Sparkles className="w-3 h-3 mr-1" />
                          Anthropic Situational Context:
                        </div>
                        {c.context_summary}
                      </div>
                    )}
                    <div className="text-slate-300 bg-slate-900/50 p-2.5 rounded border border-slate-800 text-[11px] font-mono leading-relaxed">
                      {c.raw_text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Redis Queue Telemetry Card */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-xs uppercase tracking-wider text-rose-400 flex items-center">
                <span className="w-2 h-2 rounded-full mr-2 bg-rose-500 animate-pulse"></span>
                Redis Queue Telemetry
              </h4>
              <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-rose-950/80 text-rose-300 border border-rose-800/50">
                {health?.redis_connected ? 'redis:7-alpine' : 'In-Memory Fallback'}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                <div className="text-lg font-bold text-amber-400 font-mono">
                  {health?.queue_pending_count ?? 0}
                </div>
                <div className="text-[10px] text-slate-400">Waiting</div>
              </div>
              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                <div className="text-lg font-bold text-sky-400 font-mono">
                  {health?.queue_processing_count ?? 0}
                </div>
                <div className="text-[10px] text-slate-400">In-Flight</div>
              </div>
              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                <div className="text-lg font-bold text-emerald-400 font-mono">
                  {health?.queue_completed_count ?? 0}
                </div>
                <div className="text-[10px] text-slate-400">Completed</div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-2xl border border-slate-800 text-xs text-slate-400 space-y-2.5">
            <h4 className="font-semibold text-slate-200 flex items-center">
              <BookOpen className="w-4 h-4 mr-1.5 text-indigo-400" />
              2-Step Ingestion Pipeline:
            </h4>
            <p>
              <strong className="text-emerald-400">1. Instant Raw File Cache:</strong> Uploaded files are immediately saved to disk cache within milliseconds, preventing client HTTP request timeouts on large files.
            </p>
            <p>
              <strong className="text-indigo-400">2. Background Task Queue:</strong> A background worker pulls jobs from an internal queue, executing text parsing, semantic chunking, and Anthropic contextualization.
            </p>
            <p>
              <strong className="text-purple-400">3. Live Dashboard Updates:</strong> The dashboard monitors task status through every stage (Parsing &rarr; Chunking &rarr; Contextualizing &rarr; Embedding &rarr; Indexing).
            </p>
            <p>
              <strong className="text-sky-400">4. Dual Indexing:</strong> Dense vectors stored in Qdrant & sparse tokens indexed in BM25Plus with automatic document aggregation.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}

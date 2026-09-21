import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, Sparkles, BookOpen } from 'lucide-react';
import { executeIngestApi } from '../api';

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

export default function IngestStudio({ onIngestSuccess, showToast }) {
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('');
  const [text, setText] = useState('');
  const [collection, setCollection] = useState('documents');
  const [strategy, setStrategy] = useState('semantic');
  const [useContextual, setUseContextual] = useState(true);
  const [provider, setProvider] = useState('heuristic');

  const [loading, setLoading] = useState(false);
  const [sampleChunks, setSampleChunks] = useState(null);

  const loadPreset = (key) => {
    const preset = PRESETS[key];
    if (!preset) return;
    setTitle(preset.title);
    setCategory(preset.category);
    setText(preset.text);
    showToast(`Loaded preset: ${preset.title}`);
  };

  const handleIngest = async () => {
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
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Form Column */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-bold text-base text-slate-100 flex items-center">
                <UploadCloud className="w-5 h-5 mr-2 text-indigo-400" />
                Document Ingestion & Chunking
              </h3>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-slate-400">Load sample:</span>
                <button
                  onClick={() => loadPreset('robotics')}
                  className="px-2.5 py-1 rounded text-xs bg-indigo-950/80 text-indigo-300 border border-indigo-800/40 hover:bg-indigo-900/60 transition"
                >
                  Acme Robotics
                </button>
                <button
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
                rows="9"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste full document text here. The engine will detect semantic breakpoints and prepend Anthropic situational context summaries..."
                className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

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
                <span>Ingest Document</span>
              </button>
            </div>
          </div>
        </div>

        {/* Preview / Info Column */}
        <div className="lg:col-span-1 space-y-4">
          {sampleChunks && (
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3 animate-fade-in">
              <h4 className="font-bold text-xs uppercase tracking-wider text-indigo-400 flex items-center justify-between">
                <span>Ingested Chunks Preview</span>
                <span className="text-[10px] text-slate-400">Sample</span>
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
                    <div className="text-slate-300 bg-slate-900/50 p-2.5 rounded border border-slate-800">
                      {c.raw_text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="glass-panel p-5 rounded-2xl border border-slate-800 text-xs text-slate-400 space-y-2.5">
            <h4 className="font-semibold text-slate-200 flex items-center">
              <BookOpen className="w-4 h-4 mr-1.5 text-indigo-400" />
              How Ingestion Works:
            </h4>
            <p>
              <strong className="text-indigo-400">1. Semantic Chunking:</strong> Evaluates adjacent sentence cosine distances using <code className="text-slate-300">bge-small</code> to determine topic transitions.
            </p>
            <p>
              <strong className="text-purple-400">2. Anthropic Contextual Retrieval:</strong> Generates a situational summary describing what the chunk is about in relation to the whole document.
            </p>
            <p>
              <strong className="text-sky-400">3. Dual Indexing:</strong> Dense vectors stored in Qdrant & sparse tokens indexed in BM25Plus.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}


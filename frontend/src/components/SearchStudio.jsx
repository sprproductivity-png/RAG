import React, { useState } from 'react';
import { Search, Sliders, Sparkles, Clock, Layers, ArrowUpRight } from 'lucide-react';
import { executeSearchApi, executeAskApi } from '../api';

export default function SearchStudio({ collections, showToast }) {
  const [query, setQuery] = useState('How long do the batteries last and what is the recharge time?');
  const [collection, setCollection] = useState('documents');
  const [topK, setTopK] = useState(5);
  const [denseTopK, setDenseTopK] = useState(15);
  const [sparseTopK, setSparseTopK] = useState(15);
  const [rrfK, setRrfK] = useState(60);
  const [denseWeight, setDenseWeight] = useState(0.5);
  const [sparseWeight, setSparseWeight] = useState(0.5);
  const [fusionMethod, setFusionMethod] = useState('rrf');
  const [applyReranking, setApplyReranking] = useState(true);

  const [askProvider, setAskProvider] = useState('local');
  const [askResult, setAskResult] = useState(null);
  const [isAsking, setIsAsking] = useState(false);

  const [loading, setLoading] = useState(false);
  const [searchResult, setSearchResult] = useState(null);

  const handleAsk = async (overrideQuery) => {
    const q = overrideQuery || query;
    if (!q.trim()) {
      showToast('Please enter a query', true);
      return;
    }

    setIsAsking(true);
    setSearchResult(null);
    setAskResult(null);
    try {
      const data = await executeAskApi({
        query: q,
        collection_name: collection,
        top_k: parseInt(topK),
        dense_top_k: parseInt(denseTopK),
        sparse_top_k: parseInt(sparseTopK),
        rrf_k: parseInt(rrfK),
        dense_weight: parseFloat(denseWeight),
        sparse_weight: parseFloat(sparseWeight),
        fusion_method: fusionMethod,
        enable_reranking: applyReranking,
        provider: askProvider,
        temperature: 0.1,
      });
      setAskResult(data);
      setSearchResult({
        timing_ms: { total_ms: data.duration_ms },
        total_fused_candidates: data.sources.length,
        total_dense_candidates: 0,
        total_sparse_candidates: 0,
        results: data.sources
      });
      showToast('Answer generated successfully!');
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setIsAsking(false);
    }
  };

  const handleSearch = async (overrideQuery) => {
    const q = overrideQuery || query;
    if (!q.trim()) {
      showToast('Please enter a query', true);
      return;
    }

    setLoading(true);
    setAskResult(null);
    try {
      const data = await executeSearchApi({
        query: q,
        collection_name: collection,
        top_k: parseInt(topK),
        dense_top_k: parseInt(denseTopK),
        sparse_top_k: parseInt(sparseTopK),
        rrf_k: parseInt(rrfK),
        dense_weight: parseFloat(denseWeight),
        sparse_weight: parseFloat(sparseWeight),
        fusion_method: fusionMethod,
        apply_reranking: applyReranking,
      });
      setSearchResult(data);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

        {/* Left Tuning Controls Column */}
        <div className="lg:col-span-1 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="font-semibold text-sm text-slate-200 flex items-center justify-between">
              <span className="flex items-center">
                <Sliders className="w-4 h-4 mr-1.5 text-indigo-400" />
                Retrieval Parameters
              </span>
              <span className="text-[10px] text-indigo-400 uppercase tracking-wider font-mono">Tunable</span>
            </h3>

            {/* Collection Select */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Target Collection</label>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                {collections.length === 0 ? (
                  <option value="documents">documents (default)</option>
                ) : (
                  collections.map((c) => (
                    <option key={c.collection_name} value={c.collection_name}>
                      {c.collection_name} ({c.points_count} chunks)
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* Top-K & Fusion Method */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Final Top-K</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                >
                  <option value="3">3</option>
                  <option value="5">5</option>
                  <option value="10">10</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Fusion Method</label>
                <select
                  value={fusionMethod}
                  onChange={(e) => setFusionMethod(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                >
                  <option value="rrf">RRF (Rank)</option>
                  <option value="rsf">RSF (Score)</option>
                </select>
              </div>
            </div>

            {/* Dense & Sparse Pools */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Dense Pool</label>
                <input
                  type="number"
                  value={denseTopK}
                  min="1"
                  max="100"
                  onChange={(e) => setDenseTopK(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Sparse Pool</label>
                <input
                  type="number"
                  value={sparseTopK}
                  min="1"
                  max="100"
                  onChange={(e) => setSparseTopK(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
                />
              </div>
            </div>

            {/* RRF Constant */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">RRF Constant (k=60)</label>
              <input
                type="number"
                value={rrfK}
                min="1"
                max="200"
                onChange={(e) => setRrfK(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
              />
            </div>

            {/* Dense Weight Slider */}
            <div className="pt-2 border-t border-slate-800">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-sky-400 font-medium">Dense Weight</span>
                <span className="font-mono text-slate-300">{parseFloat(denseWeight).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={denseWeight}
                onChange={(e) => setDenseWeight(e.target.value)}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            {/* Sparse Weight Slider */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-amber-400 font-medium">BM25 Sparse Weight</span>
                <span className="font-mono text-slate-300">{parseFloat(sparseWeight).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={sparseWeight}
                onChange={(e) => setSparseWeight(e.target.value)}
                className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            {/* Cross-Encoder Toggle */}
            <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
              <div>
                <span className="block text-xs font-semibold text-purple-300">Cross-Encoder</span>
                <span className="text-[10px] text-slate-400">Full self-attention re-rank</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={applyReranking}
                  onChange={(e) => setApplyReranking(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
              </label>
            </div>
          </div>
        </div>

        {/* Right Search Input & Results Column */}
        <div className="lg:col-span-3 space-y-4">
          {/* Query Bar */}
          <div className="glass-panel p-3.5 rounded-2xl border border-slate-800 shadow-xl">
            <div className="flex items-center space-x-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                  placeholder="Ask a question or enter keywords to retrieve..."
                  className="w-full bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <button
                onClick={() => handleAsk()}
                disabled={loading}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 transition flex items-center disabled:opacity-50"
              >
                {loading ? <div className="spinner mr-2"></div> : <Search className="w-4 h-4 mr-1.5" />}
                <span>Search</span>
              </button>
            </div>

            {/* Quick-fill query chips */}
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
              <span className="font-medium text-slate-500">Sample queries:</span>
              <button
                onClick={() => {
                  const q = 'How long do the batteries last and what is the recharge time?';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                Battery & Charging
              </button>
              <button
                onClick={() => {
                  const q = 'How do the robots prevent collisions and deadlocks in narrow aisles?';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                Path Deadlocks
              </button>
              <button
                onClick={() => {
                  const q = 'Explain the SLAM positioning accuracy and sensors.';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                SLAM Accuracy
              </button>
            </div>
          </div>

          {/* AI Answer & Sources */}
          {searchResult && (
            <div className="space-y-4">
              {/* Latency Banner */}
              <div className="glass-panel rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-indigo-400" />
                  <span className="font-semibold text-slate-100">Latency:</span>
                  <span className="px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 font-mono font-bold">
                    {searchResult.timing_ms?.total_ms} ms
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-4 text-slate-400">
                  <span>Dense: <strong className="text-sky-400 font-mono">{searchResult.timing_ms?.dense_retrieval_ms}ms</strong></span>
                  <span>BM25: <strong className="text-amber-400 font-mono">{searchResult.timing_ms?.sparse_retrieval_ms}ms</strong></span>
                  <span>Fusion: <strong className="text-emerald-400 font-mono">{searchResult.timing_ms?.fusion_ms}ms</strong></span>
                  <span>Re-ranker: <strong className="text-purple-400 font-mono">{searchResult.timing_ms?.reranking_ms}ms</strong></span>
                </div>
                <div className="text-slate-400 text-[11px]">
                  Candidates: {searchResult.total_fused_candidates} fused ({searchResult.total_dense_candidates} dense, {searchResult.total_sparse_candidates} sparse)
                </div>
              </div>

              {/* Result Cards */}
              {searchResult.results.length === 0 ? (
                <div className="text-center py-12 glass-panel rounded-2xl border border-slate-800">
                  <p className="text-slate-400 text-sm">No matching documents found.</p>
                  <p className="text-slate-500 text-xs mt-1">Try ingesting documents in the Ingestion Studio tab first.</p>
                </div>
              ) : (
                searchResult.results.map((r, idx) => {
                  const ceScore = r.cross_encoder_score !== null ? r.cross_encoder_score.toFixed(4) : 'N/A';
                  const denseScore = r.dense_score !== null ? r.dense_score.toFixed(4) : 'N/A';
                  const bm25Score = r.bm25_score !== null ? r.bm25_score.toFixed(4) : 'N/A';
                  const fusionScore = r.fusion_score !== null ? r.fusion_score.toFixed(6) : 'N/A';

                  return (
                    <div key={r.chunk_id || idx} className="glass-panel card-hover rounded-xl p-5 border border-slate-800 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center space-x-2.5">
                          <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-600/30 text-indigo-300 font-bold text-sm border border-indigo-500/30">
                            #{idx + 1}
                          </span>
                          <span className="text-xs text-slate-400 font-mono">{r.chunk_id?.slice(0, 8)}...</span>
                          {r.metadata?.title && (
                            <span className="text-xs font-medium text-slate-300 bg-slate-800 px-2 py-0.5 rounded">
                              {r.metadata.title}
                            </span>
                          )}
                        </div>

                        {/* Score Badges */}
                        <div className="flex flex-wrap items-center gap-2">
                          {r.cross_encoder_score !== null && (
                            <div className="badge-cross-encoder px-2.5 py-1 rounded-md text-xs font-semibold flex items-center shadow-sm">
                              <span>Cross-Encoder: <strong>{ceScore}</strong></span>
                            </div>
                          )}
                          <div className="badge-dense px-2 py-0.5 rounded text-xs">
                            Dense: <strong>{denseScore}</strong> {r.dense_rank ? `(#${r.dense_rank})` : ''}
                          </div>
                          <div className="badge-bm25 px-2 py-0.5 rounded text-xs">
                            BM25: <strong>{bm25Score}</strong> {r.bm25_rank ? `(#${r.bm25_rank})` : ''}
                          </div>
                          <div className="badge-rrf px-2 py-0.5 rounded text-xs">
                            RRF: <strong>{fusionScore}</strong>
                          </div>
                        </div>
                      </div>

                      {/* Anthropic Contextual Retrieval Summary Box */}
                      {r.context_summary && (
                        <div className="badge-context rounded-lg p-3 text-xs leading-relaxed">
                          <div className="flex items-center font-semibold mb-1 text-purple-300">
                            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
                            Anthropic Contextual Retrieval Summary
                          </div>
                          <p className="text-purple-200/90">{r.context_summary}</p>
                        </div>
                      )}

                      {/* Raw Chunk Passage */}
                      <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap bg-slate-900/40 p-3 rounded-lg border border-slate-800/60 font-normal">
                        {r.text}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {!searchResult && (
            <div className="text-center py-16 glass-panel rounded-2xl border border-slate-800">
              <div className="w-12 h-12 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 flex items-center justify-center mx-auto mb-3">
                <Search className="w-6 h-6" />
              </div>
              <h4 className="font-semibold text-slate-300">Ready to Ask</h4>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Ask questions and get AI-generated answers based strictly on your source documents.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


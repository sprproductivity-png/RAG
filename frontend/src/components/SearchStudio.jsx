import React, { useState } from 'react';
import { Search, Sliders, Sparkles, Clock, Layers, ArrowUpRight, Copy, Check, BookOpen, Bot } from 'lucide-react';
import { executeSearchApi, executeAskApi } from '../api';

function renderInline(str) {
  if (!str) return null;
  // Splits into **bold**, `code`, and [1] or [Source 1] citations
  const tokens = str.split(/(\*\*.*?\*\*|`[^`]+`|\[(?:Source\s*)?\d+(?:,\s*\d+)*\])/g);
  return tokens.map((token, i) => {
    if (token.startsWith('**') && token.endsWith('**')) {
      return <strong key={i} className="text-white font-semibold">{token.slice(2, -2)}</strong>;
    }
    if (token.startsWith('`') && token.endsWith('`')) {
      return <code key={i} className="px-1.5 py-0.5 rounded bg-slate-800 text-indigo-300 font-mono text-xs">{token.slice(1, -1)}</code>;
    }
    if (/^\[(?:Source\s*)?\d+(?:,\s*\d+)*\]$/.test(token)) {
      return (
        <span
          key={i}
          className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-mono font-bold bg-indigo-900/80 text-indigo-300 border border-indigo-700/60 shadow-sm cursor-default"
          title="Source citation"
        >
          {token}
        </span>
      );
    }
    return token;
  });
}

function FormattedAnswer({ text }) {
  if (!text) return null;

  const paragraphs = text.split(/\n\s*\n/);

  return (
    <div className="space-y-3.5 text-slate-100 text-sm leading-relaxed">
      {paragraphs.map((para, pIdx) => {
        const trimmed = para.trim();
        if (!trimmed) return null;

        if (trimmed.startsWith('### ')) {
          return (
            <h4 key={pIdx} className="text-base font-bold text-indigo-200 mt-4 mb-2 flex items-center">
              <span className="w-1.5 h-4 bg-indigo-500 rounded mr-2 inline-block"></span>
              {renderInline(trimmed.replace(/^###\s+/, ''))}
            </h4>
          );
        }
        if (trimmed.startsWith('## ')) {
          return (
            <h3 key={pIdx} className="text-lg font-bold text-white mt-5 mb-2 pb-1 border-b border-slate-800">
              {renderInline(trimmed.replace(/^##\s+/, ''))}
            </h3>
          );
        }
        if (trimmed.startsWith('# ')) {
          return (
            <h2 key={pIdx} className="text-xl font-extrabold text-white mt-6 mb-3">
              {renderInline(trimmed.replace(/^#\s+/, ''))}
            </h2>
          );
        }

        const lines = trimmed.split('\n');
        const isList = lines.length > 0 && lines.every(l => l.trim().startsWith('- ') || l.trim().startsWith('* ') || /^\d+\.\s/.test(l.trim()));
        if (isList) {
          return (
            <ul key={pIdx} className="space-y-1.5 pl-4 list-disc list-outside marker:text-indigo-400">
              {lines.map((line, lIdx) => {
                const cleanLine = line.trim().replace(/^[-*]\s+|\d+\.\s+/, '');
                return <li key={lIdx}>{renderInline(cleanLine)}</li>;
              })}
            </ul>
          );
        }

        return <p key={pIdx}>{renderInline(trimmed)}</p>;
      })}
    </div>
  );
}

function formatChunkText(raw) {
  if (!raw) return '';
  const lines = raw.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return raw;
  const avgWords = lines.reduce((acc, l) => acc + l.split(/\s+/).length, 0) / lines.length;
  if (avgWords < 3.5) {
    return lines
      .map(l => (l.startsWith('●') || l.startsWith('---') || l.startsWith('-') || /^\d+\./.test(l)) ? `\n\n${l}` : l)
      .join(' ')
      .replace(/\s+/g, ' ')
      .replace(/\n\s+/g, '\n')
      .trim();
  }
  return raw;
}

export default function SearchStudio({ collections, showToast }) {
  const [query, setQuery] = useState('What are the key concepts of deep learning and neural networks?');
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
  const [copied, setCopied] = useState(false);

  const handleAsk = async (overrideQuery) => {
    const q = overrideQuery || query;
    if (!q.trim()) {
      showToast('Please enter a query', true);
      return;
    }

    setIsAsking(true);
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
        apply_reranking: applyReranking,
        provider: askProvider,
        temperature: 0.1,
      });
      setAskResult(data);
      showToast('Answer generated successfully!');
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setIsAsking(false);
    }
  };

  const handleCopyAnswer = () => {
    if (!askResult?.answer) return;
    navigator.clipboard.writeText(askResult.answer);
    setCopied(true);
    showToast('Answer copied to clipboard!');
    setTimeout(() => setCopied(false), 2000);
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
                Retrieval & Model
              </span>
              <span className="text-[10px] text-indigo-400 uppercase tracking-wider font-mono">NotebookLM</span>
            </h3>

            {/* Collection Select */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Target Collection</label>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-medium"
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

            {/* LLM Provider Select */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">LLM Provider</label>
              <select
                value={askProvider}
                onChange={(e) => setAskProvider(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-indigo-300 font-semibold focus:outline-none focus:border-indigo-500"
              >
                <option value="local">Local Qwen-1.5B (Fast Offline GGUF)</option>
                <option value="ollama">Ollama (AMD GPU / Local Server)</option>
                <option value="anthropic">Anthropic Claude (Cloud)</option>
                <option value="openai">OpenAI GPT-4o-mini (Cloud)</option>
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
                  <option value="3">3 Chunks</option>
                  <option value="5">5 Chunks</option>
                  <option value="8">8 Chunks</option>
                  <option value="10">10 Chunks</option>
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
        <div className="lg:col-span-3 space-y-5">
          {/* Query Bar */}
          <div className="glass-panel p-3.5 rounded-2xl border border-slate-800 shadow-xl">
            <div className="flex items-center space-x-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !isAsking && handleAsk()}
                  placeholder="Ask any question across your uploaded documents..."
                  className="w-full bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <button
                onClick={() => handleAsk()}
                disabled={isAsking}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 transition flex items-center disabled:opacity-50"
              >
                {isAsking ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2"></div>
                    <span>Synthesizing...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    <span>Ask AI</span>
                  </>
                )}
              </button>
            </div>

            {/* Quick-fill query chips */}
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
              <span className="font-medium text-slate-500">Sample questions:</span>
              <button
                onClick={() => {
                  const q = 'What is deep learning and how do multi-layer neural networks work?';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                Deep Learning Foundations
              </button>
              <button
                onClick={() => {
                  const q = 'How does backpropagation compute gradients with the chain rule?';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                Backpropagation Gradients
              </button>
              <button
                onClick={() => {
                  const q = 'Explain how the Transformer attention mechanism operates.';
                  setQuery(q);
                  handleAsk(q);
                }}
                className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition"
              >
                Transformer Attention
              </button>
            </div>
          </div>

          {/* Loading Indicator */}
          {isAsking && (
            <div className="glass-panel rounded-2xl p-8 border border-indigo-500/30 text-center space-y-3 shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-indigo-900/40 border border-indigo-700/40 text-indigo-400 flex items-center justify-center mx-auto animate-pulse">
                <Sparkles className="w-6 h-6 animate-spin" />
              </div>
              <h4 className="font-semibold text-slate-100 text-base">Synthesizing NotebookLM Answer</h4>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Executing dense embedding & BM25 hybrid fusion, re-ranking top passages with Cross-Encoder, and generating a grounded response with {askProvider.toUpperCase()}...
              </p>
            </div>
          )}

          {/* AI Synthesized Answer Card */}
          {askResult && !isAsking && (
            <div className="space-y-6">
              <div className="glass-panel rounded-2xl border border-indigo-500/30 p-6 shadow-2xl relative overflow-hidden bg-gradient-to-b from-indigo-950/20 to-slate-900/60">
                <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
                  <div className="flex items-center space-x-2.5">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-md">
                      <Sparkles className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-100 text-sm">AI Synthesized Answer</h4>
                      <p className="text-[11px] text-slate-400">Strictly grounded in {askResult.sources?.length || 0} referenced document passages</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className="text-xs font-mono font-medium text-slate-400 flex items-center bg-slate-800/80 px-2.5 py-1 rounded-md">
                      <Clock className="w-3.5 h-3.5 mr-1.5 text-indigo-400" />
                      {askResult.duration_ms} ms
                    </span>
                    <button
                      onClick={handleCopyAnswer}
                      className="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs flex items-center space-x-1.5 transition"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                </div>

                {/* Formatted Markdown Answer */}
                <div className="py-1">
                  <FormattedAnswer text={askResult.answer} />
                </div>
              </div>

              {/* Grounding Sources Section */}
              <div className="space-y-3">
                <div className="flex items-center justify-between px-1">
                  <h4 className="font-bold text-sm text-slate-200 flex items-center">
                    <BookOpen className="w-4 h-4 mr-2 text-indigo-400" />
                    Grounding Sources ({askResult.sources?.length || 0} Citations)
                  </h4>
                  <span className="text-xs text-slate-500 font-mono">Ranked by Cross-Encoder</span>
                </div>

                {askResult.sources?.map((r, idx) => {
                  const ceScore = r.cross_encoder_score !== null ? r.cross_encoder_score.toFixed(4) : 'N/A';
                  const denseScore = r.dense_score !== null ? r.dense_score.toFixed(4) : 'N/A';
                  const bm25Score = r.bm25_score !== null ? r.bm25_score.toFixed(4) : 'N/A';

                  return (
                    <div key={r.chunk_id || idx} className="glass-panel rounded-xl p-4 border border-slate-800/80 space-y-2.5">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center space-x-2">
                          <span className="px-2 py-0.5 rounded-md bg-indigo-600/30 text-indigo-300 font-mono font-bold text-xs border border-indigo-500/30">
                            Source [{idx + 1}]
                          </span>
                          <span className="text-xs font-semibold text-slate-200">
                            {r.metadata?.title || `Document ${r.doc_id}`}
                          </span>
                          {r.metadata?.page_count && (
                            <span className="text-[11px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                              Page {r.metadata.page_count}
                            </span>
                          )}
                        </div>

                        {/* Relevance Scores */}
                        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                          {r.cross_encoder_score !== null && (
                            <span className="px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 font-mono font-semibold border border-purple-800/40">
                              Relevance: {ceScore}
                            </span>
                          )}
                          <span className="px-2 py-0.5 rounded bg-sky-950/80 text-sky-300 font-mono border border-sky-800/40">
                            Dense: {denseScore}
                          </span>
                          <span className="px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 font-mono border border-amber-800/40">
                            BM25: {bm25Score}
                          </span>
                        </div>
                      </div>

                      {/* Anthropic Contextual Summary */}
                      {r.context_summary && (
                        <div className="badge-context rounded-lg p-2.5 text-xs text-purple-200/90 flex items-start space-x-2">
                          <Sparkles className="w-3.5 h-3.5 text-purple-400 mt-0.5 shrink-0" />
                          <span>{r.context_summary}</span>
                        </div>
                      )}

                      {/* Chunk Content */}
                      <div className="text-slate-300 text-xs leading-relaxed whitespace-pre-line bg-slate-900/50 p-3 rounded-lg border border-slate-800/50">
                        {formatChunkText(r.text)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {!askResult && !isAsking && (
            <div className="text-center py-16 glass-panel rounded-2xl border border-slate-800">
              <div className="w-12 h-12 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 flex items-center justify-center mx-auto mb-3">
                <Bot className="w-6 h-6" />
              </div>
              <h4 className="font-semibold text-slate-300">NotebookLM Generative QA</h4>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Type any question above to synthesize a grounded, cited answer from your indexed documents.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

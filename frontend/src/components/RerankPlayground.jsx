import React, { useState } from 'react';
import { Layers, Play, CheckCircle2 } from 'lucide-react';
import { executeRerankApi } from '../api';

export default function RerankPlayground({ showToast }) {
  const [query, setQuery] = useState('What battery technology enables 18 hours of operation?');
  const [p1, setP1] = useState(
    'Quantum entanglement occurs when two particles remain physically linked regardless of distance in spacetime.'
  );
  const [p2, setP2] = useState(
    'The Falcon-5 powertrain integrates lithium iron phosphate (LiFePO4) battery packs delivering 18 hours of continuous operation with 35-minute fast charging.'
  );
  const [p3, setP3] = useState(
    'Our mobile robots utilize graph-based SLAM with high-frequency LiDAR scans for autonomous warehouse navigation.'
  );

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleRerank = async () => {
    if (!query.trim()) {
      showToast('Please enter a query', true);
      return;
    }

    const passages = [p1, p2, p3].filter((p) => p.trim().length > 0);
    if (passages.length === 0) {
      showToast('Please enter at least one passage', true);
      return;
    }

    setLoading(true);
    try {
      const data = await executeRerankApi({
        query,
        documents: passages.map((text, idx) => ({ id: `cand_${idx + 1}`, text })),
      });
      setResults(data.results);
      showToast(`Re-ranking complete in ${data.duration_ms}ms`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div>
          <h3 className="font-bold text-base text-slate-100 flex items-center">
            <Layers className="w-5 h-5 mr-2 text-purple-400" />
            Cross-Encoder Re-ranking Playground
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Test full joint self-attention across query and candidate passages using{' '}
            <code className="text-purple-300">cross-encoder/ms-marco-MiniLM-L-6-v2</code>.
          </p>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Test Query</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div className="space-y-3">
          <label className="block text-xs font-medium text-slate-400">Candidate Passages to Compare</label>
          <div>
            <span className="text-[11px] text-slate-500 mb-1 block">Passage 1 (Irrelevant):</span>
            <textarea
              rows="2"
              value={p1}
              onChange={(e) => setP1(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono"
            />
          </div>
          <div>
            <span className="text-[11px] text-slate-500 mb-1 block">Passage 2 (Highly Relevant):</span>
            <textarea
              rows="2"
              value={p2}
              onChange={(e) => setP2(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono"
            />
          </div>
          <div>
            <span className="text-[11px] text-slate-500 mb-1 block">Passage 3 (Partially Relevant):</span>
            <textarea
              rows="2"
              value={p3}
              onChange={(e) => setP3(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono"
            />
          </div>
        </div>

        <button
          onClick={handleRerank}
          disabled={loading}
          className="px-6 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/25 transition flex items-center disabled:opacity-50"
        >
          {loading ? <div className="spinner mr-2"></div> : <Play className="w-4 h-4 mr-1.5" />}
          <span>Score & Re-rank Passages</span>
        </button>
      </div>

      {/* Results Output */}
      {results && (
        <div className="space-y-3">
          <h4 className="font-semibold text-xs text-slate-300 uppercase tracking-wider">
            Re-ranked Order (Highest Relevance First):
          </h4>
          {results.map((r, rank) => (
            <div key={r.id || rank} className="glass-panel card-hover p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="flex items-center justify-center w-6 h-6 rounded bg-purple-600/30 text-purple-300 font-bold text-xs">
                  #{rank + 1}
                </span>
                <span className="badge-cross-encoder px-2.5 py-1 rounded text-xs font-semibold">
                  Cross-Encoder Logit: {r.score}
                </span>
              </div>
              <div className="text-slate-200 text-xs leading-relaxed bg-slate-900/40 p-3 rounded">
                {r.text}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


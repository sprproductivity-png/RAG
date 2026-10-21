import React from 'react';
import { CheckCircle2, Clock, AlertTriangle, Loader2, Sparkles, Layers, FileText, Database, ShieldCheck, RefreshCw } from 'lucide-react';

const STAGES = [
  { id: 'queued', label: '1. Cached', icon: Clock },
  { id: 'parsing', label: '2. Text Parsing', icon: FileText },
  { id: 'chunking', label: '3. Semantic Chunking', icon: Layers },
  { id: 'contextualizing', label: '4. Situational Context', icon: Sparkles },
  { id: 'embedding', label: '5. Vector Embedding', icon: Database },
  { id: 'indexing', label: '6. Dual Indexing', icon: Database },
];

export default function TaskProgressCard({ task, onClear, onRetry }) {
  if (!task) return null;

  const isCompleted = task.status === 'completed';
  const isFailed = task.status === 'failed';
  const isRunning = task.status === 'processing' || task.status === 'queued';

  const getStageIndex = (stage) => {
    switch (stage) {
      case 'queued': return 0;
      case 'parsing': return 1;
      case 'chunking': return 2;
      case 'contextualizing': return 3;
      case 'embedding': return 4;
      case 'indexing': return 5;
      case 'completed': return 6;
      default: return 0;
    }
  };

  const currentIdx = getStageIndex(task.stage);

  return (
    <div className={`p-5 rounded-2xl border transition-all duration-300 ${
      isCompleted
        ? 'bg-emerald-950/20 border-emerald-800/40'
        : isFailed
        ? 'bg-rose-950/20 border-rose-800/40'
        : 'bg-indigo-950/20 border-indigo-800/40 animate-pulse-border'
    }`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center space-x-2.5">
          {isRunning && <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />}
          {isCompleted && <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
          {isFailed && <AlertTriangle className="w-5 h-5 text-rose-400" />}
          <div>
            <h4 className="font-bold text-sm text-slate-100 flex items-center space-x-2">
              <span>{task.filename}</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                {task.task_id}
              </span>
            </h4>
            <div className="text-[11px] text-slate-400 mt-0.5 flex items-center space-x-3">
              <span>Target: <strong className="text-indigo-300 font-mono">{task.collection_name}</strong></span>
              {task.file_size && (
                <span>Size: <strong className="text-slate-300">{(task.file_size / 1024).toFixed(1)} KB</strong></span>
              )}
              {task.duration_ms && (
                <span>Processed in: <strong className="text-emerald-300">{task.duration_ms} ms</strong></span>
              )}
              {task.chunks_created && (
                <span>Chunks: <strong className="text-indigo-300">{task.chunks_created}</strong></span>
              )}
            </div>
          </div>
        </div>

        {/* Status Pill */}
        <div className="flex items-center space-x-2">
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border ${
            isCompleted
              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50'
              : isFailed
              ? 'bg-rose-950/60 text-rose-300 border-rose-700/50'
              : 'bg-indigo-950/60 text-indigo-300 border-indigo-700/50 animate-pulse'
          }`}>
            {task.status}
          </span>
          {isFailed && onRetry && (
            <button
              onClick={() => onRetry(task.task_id)}
              className="text-[11px] font-semibold text-amber-300 hover:text-amber-100 px-2.5 py-1 rounded bg-amber-950/80 border border-amber-800/60 hover:bg-amber-900/60 transition flex items-center space-x-1"
              title="Resume ingestion from last saved checkpoint"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              <span>Resume from Checkpoint</span>
            </button>
          )}
          {onClear && (
            <button
              onClick={onClear}
              className="text-[11px] text-slate-400 hover:text-slate-200 px-2 py-1 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 transition"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="space-y-1.5 my-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-300 font-medium">{task.stage_message}</span>
          <span className="font-mono font-semibold text-indigo-400">{task.progress}%</span>
        </div>
        <div className="w-full bg-slate-900/80 rounded-full h-2.5 overflow-hidden border border-slate-800">
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isCompleted
                ? 'bg-gradient-to-r from-emerald-600 to-teal-400'
                : isFailed
                ? 'bg-rose-600'
                : 'bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-400'
            }`}
            style={{ width: `${task.progress}%` }}
          />
        </div>
      </div>

      {/* Stage pipeline pills */}
      <div className="grid grid-cols-2 sm:grid-cols-6 gap-1.5 pt-2 text-[10px] border-t border-slate-800/80">
        {STAGES.map((s, idx) => {
          const isDone = currentIdx > idx || isCompleted;
          const isCurrent = currentIdx === idx && !isCompleted && !isFailed;
          return (
            <div
              key={s.id}
              className={`p-1.5 rounded-lg border text-center transition flex flex-col items-center justify-center space-y-0.5 ${
                isDone
                  ? 'bg-emerald-950/30 border-emerald-800/40 text-emerald-300'
                  : isCurrent
                  ? 'bg-indigo-900/40 border-indigo-600 text-indigo-200 font-semibold shadow-sm shadow-indigo-500/20'
                  : 'bg-slate-900/40 border-slate-800 text-slate-500'
              }`}
            >
              <span>{s.label}</span>
            </div>
          );
        })}
      </div>

      {/* Error Message if Failed */}
      {isFailed && task.error && (
        <div className="mt-3 p-3 rounded-xl bg-rose-950/40 border border-rose-800/50 text-xs text-rose-300 font-mono">
          {task.error}
        </div>
      )}
    </div>
  );
}

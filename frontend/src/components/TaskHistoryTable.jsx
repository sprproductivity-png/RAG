import React from 'react';
import { History, RefreshCw, Trash2, CheckCircle2, Clock, AlertTriangle, Loader2, Eye } from 'lucide-react';
import { deleteTaskApi } from '../api';

export default function TaskHistoryTable({ tasks, onRefresh, onSelectTask, onRetry, showToast }) {
  const handleDelete = async (taskId, e) => {
    e.stopPropagation();
    try {
      await deleteTaskApi(taskId);
      showToast(`Removed task ${taskId}`);
      if (onRefresh) onRefresh();
    } catch (err) {
      showToast(err.message, true);
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
            <CheckCircle2 className="w-3 h-3 mr-1" />
            Completed
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-indigo-950/60 text-indigo-300 border border-indigo-800/40 animate-pulse">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Processing
          </span>
        );
      case 'queued':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950/60 text-amber-300 border border-amber-800/40">
            <Clock className="w-3 h-3 mr-1" />
            Queued
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-950/60 text-rose-300 border border-rose-800/40">
            <AlertTriangle className="w-3 h-3 mr-1" />
            Failed
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-bold text-xs uppercase tracking-wider text-slate-300 flex items-center space-x-2">
          <History className="w-4 h-4 text-indigo-400" />
          <span>Ingestion Queue & History</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
            {tasks.length}
          </span>
        </h4>

        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-xs text-slate-400 hover:text-slate-200 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 transition flex items-center space-x-1"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Refresh</span>
          </button>
        )}
      </div>

      <div className="overflow-x-auto max-h-60 overflow-y-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="text-[10px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-800 sticky top-0">
            <tr>
              <th className="py-2 px-3">File / Task</th>
              <th className="py-2 px-3">Target Collection</th>
              <th className="py-2 px-3">Status</th>
              <th className="py-2 px-3">Chunks</th>
              <th className="py-2 px-3">Duration</th>
              <th className="py-2 px-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {tasks.length === 0 ? (
              <tr>
                <td colSpan="6" className="text-center py-6 text-slate-500 text-xs">
                  No tasks queued yet. Upload a file above to begin 2-step cached ingestion!
                </td>
              </tr>
            ) : (
              tasks.map((t) => (
                <tr
                  key={t.task_id}
                  onClick={() => onSelectTask && onSelectTask(t)}
                  className="border-b border-slate-800/60 hover:bg-slate-800/30 transition cursor-pointer"
                >
                  <td className="py-2.5 px-3">
                    <div className="font-semibold text-slate-200 truncate max-w-xs">{t.filename}</div>
                    <div className="text-[10px] font-mono text-slate-500">{t.task_id}</div>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-indigo-300">{t.collection_name}</td>
                  <td className="py-2.5 px-3">{getStatusBadge(t.status)}</td>
                  <td className="py-2.5 px-3 font-semibold text-slate-300">
                    {t.chunks_created != null ? `${t.chunks_created} chunks` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {t.duration_ms != null ? `${t.duration_ms} ms` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <div className="flex items-center justify-end space-x-1.5">
                      {t.status === 'failed' && onRetry && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onRetry(t.task_id);
                          }}
                          className="p-1 px-2 rounded text-[11px] font-semibold bg-amber-950/80 text-amber-300 border border-amber-800/50 hover:bg-amber-900/60 transition flex items-center space-x-1"
                          title="Resume task from last checkpoint"
                        >
                          <RefreshCw className="w-3 h-3 mr-1" />
                          <span>Resume</span>
                        </button>
                      )}
                      <button
                        onClick={(e) => handleDelete(t.task_id, e)}
                        className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-rose-950/30 transition"
                        title="Dismiss task"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
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

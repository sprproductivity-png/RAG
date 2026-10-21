import React, { useState, useEffect } from 'react';
import { 
  Database, Trash2, RefreshCw, Sparkles, CheckCircle2, 
  AlertTriangle, HardDrive, RotateCcw, Wrench, ShieldCheck, 
  Clock, Loader2, FileText, Activity
} from 'lucide-react';
import { deleteCollectionApi, triggerHousekeepApi, fetchHousekeepStatusApi } from '../api';

export default function CollectionsView({ collections, onRefresh, onViewDocuments, showToast }) {
  const [housekeepReport, setHousekeepReport] = useState(null);
  const [isRunningHousekeep, setIsRunningHousekeep] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);

  const loadHousekeepReport = async () => {
    setLoadingReport(true);
    try {
      const data = await fetchHousekeepStatusApi();
      setHousekeepReport(data);
    } catch (err) {
      console.warn('Could not fetch housekeeping report:', err);
    } finally {
      setLoadingReport(false);
    }
  };

  useEffect(() => {
    loadHousekeepReport();
  }, []);

  const handleRunHousekeeping = async () => {
    setIsRunningHousekeep(true);
    try {
      const report = await triggerHousekeepApi();
      setHousekeepReport(report);
      const freedMb = report?.cache_cleanup?.freed_mb ?? 0;
      const retriedCount = report?.retried_tasks?.length ?? 0;
      const reconciledCount = (report?.reconciliation?.reconciled_tasks?.length ?? 0) + 
                              (report?.reconciliation?.collections_synced?.length ?? 0);

      showToast(
        `Housekeeping completed in ${report.duration_ms}ms: ${retriedCount} task(s) retried, ${freedMb} MB disk freed, ${reconciledCount} issue(s) reconciled.`
      );
      if (onRefresh) onRefresh();
    } catch (err) {
      showToast(err.message || 'Housekeeping failed', true);
    } finally {
      setIsRunningHousekeep(false);
    }
  };

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
    <div className="space-y-6">
      {/* 1. Vector Collections & BM25 Indices Panel */}
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
                    <td className="py-3 px-4 text-slate-300 font-mono">{c.points_count}</td>
                    <td className="py-3 px-4 text-slate-300 font-mono">{c.vector_size}</td>
                    <td className="py-3 px-4 text-slate-300">{c.distance}</td>
                    <td className="py-3 px-4 text-slate-300 font-mono">{c.bm25_docs_count}</td>
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

      {/* 2. System Maintenance & Housekeeper Console */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5 animate-fade-in">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <h3 className="font-bold text-base text-slate-100 flex items-center">
              <Wrench className="w-5 h-5 mr-2 text-emerald-400" />
              Automated Maintenance & Self-Healing Housekeeper
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Background scheduler runs automatically every 5 minutes. Retries transient failures from checkpoints, removes completed & orphaned cache files, and reconciles Qdrant & BM25 synchronization.
            </p>
          </div>

          <div className="flex items-center space-x-3 shrink-0">
            <button
              onClick={handleRunHousekeeping}
              disabled={isRunningHousekeep}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-semibold shadow-lg shadow-emerald-900/30 transition flex items-center space-x-2 disabled:opacity-50"
            >
              {isRunningHousekeep ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Running Maintenance...</span>
                </>
              ) : (
                <>
                  <RotateCcw className="w-4 h-4 text-emerald-200" />
                  <span>Run Housekeeping Now</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Status Metrics Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {/* Card 1: Status & Duration */}
          <div className="bg-slate-900/70 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center">
                <ShieldCheck className="w-4 h-4 mr-1.5 text-emerald-400" />
                Service Health
              </span>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            </div>
            <div className="mt-2">
              <div className="text-lg font-bold text-emerald-300 font-mono uppercase">
                {housekeepReport?.status || 'Active'}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5 flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                Cycle: {housekeepReport?.duration_ms ? `${housekeepReport.duration_ms} ms` : '5 min loop'}
              </div>
            </div>
          </div>

          {/* Card 2: Auto-Retried Tasks */}
          <div className="bg-slate-900/70 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center">
                <RotateCcw className="w-4 h-4 mr-1.5 text-sky-400" />
                Tasks Auto-Retried
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-sky-950 text-sky-300 border border-sky-800/60 font-mono">
                From Checkpoint
              </span>
            </div>
            <div className="mt-2">
              <div className="text-lg font-bold text-sky-300 font-mono">
                {housekeepReport?.retried_tasks?.length ?? 0}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Transient timeouts / restart recovery
              </div>
            </div>
          </div>

          {/* Card 3: Cache Cleanup */}
          <div className="bg-slate-900/70 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center">
                <HardDrive className="w-4 h-4 mr-1.5 text-amber-400" />
                Disk Freed
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-950 text-amber-300 border border-amber-800/60 font-mono">
                {housekeepReport?.cache_cleanup?.deleted_files ?? 0} files
              </span>
            </div>
            <div className="mt-2">
              <div className="text-lg font-bold text-amber-300 font-mono">
                {housekeepReport?.cache_cleanup?.freed_mb ?? 0} MB
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Stale raw files & finished checkpoints
              </div>
            </div>
          </div>

          {/* Card 4: Integrity Reconciled */}
          <div className="bg-slate-900/70 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center">
                <Activity className="w-4 h-4 mr-1.5 text-purple-400" />
                Data Integrity
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-950 text-purple-300 border border-purple-800/60 font-mono">
                Synced
              </span>
            </div>
            <div className="mt-2">
              <div className="text-lg font-bold text-purple-300 font-mono">
                {(housekeepReport?.reconciliation?.reconciled_tasks?.length ?? 0) + 
                 (housekeepReport?.reconciliation?.collections_synced?.length ?? 0)}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Stalled tasks healed & BM25 synced
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Audit Logs if any actions were taken */}
        {housekeepReport && (
          <div className="space-y-3 pt-2">
            {/* Retried Tasks details */}
            {housekeepReport.retried_tasks && housekeepReport.retried_tasks.length > 0 && (
              <div className="bg-slate-900/50 p-3 rounded-xl border border-sky-800/30 text-xs space-y-2">
                <div className="font-semibold text-sky-300 flex items-center">
                  <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                  Auto-Retried Ingestion Tasks:
                </div>
                <div className="space-y-1.5">
                  {housekeepReport.retried_tasks.map((r, i) => (
                    <div key={i} className="flex items-center justify-between bg-slate-950/60 p-2 rounded border border-slate-800 font-mono text-[11px]">
                      <span className="text-slate-200 font-semibold">{r.filename}</span>
                      <span className="text-slate-400">({r.task_id})</span>
                      <span className="text-amber-400">Attempt {r.attempt}</span>
                      <span className="text-slate-500 truncate max-w-xs">{r.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reconciled Collections details */}
            {housekeepReport.reconciliation?.collections_synced && housekeepReport.reconciliation.collections_synced.length > 0 && (
              <div className="bg-slate-900/50 p-3 rounded-xl border border-purple-800/30 text-xs space-y-2">
                <div className="font-semibold text-purple-300 flex items-center">
                  <Activity className="w-3.5 h-3.5 mr-1.5" />
                  Reconciled Qdrant & BM25 Collections:
                </div>
                <div className="space-y-1.5">
                  {housekeepReport.reconciliation.collections_synced.map((c, i) => (
                    <div key={i} className="flex items-center justify-between bg-slate-950/60 p-2 rounded border border-slate-800 font-mono text-[11px]">
                      <span className="text-indigo-300 font-semibold">{c.collection}</span>
                      <span className="text-slate-400">{c.points_in_qdrant} points in Qdrant</span>
                      <span className="text-emerald-400">Synced {c.synced_bm25_count} to BM25</span>
                      <span className="text-slate-500">{c.action}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cleaned cache summary */}
            {housekeepReport.cache_cleanup?.cleaned && housekeepReport.cache_cleanup.cleaned.length > 0 && (
              <div className="bg-slate-900/50 p-3 rounded-xl border border-slate-800 text-xs space-y-1.5">
                <div className="font-semibold text-slate-300 flex items-center">
                  <HardDrive className="w-3.5 h-3.5 mr-1.5 text-amber-400" />
                  Cleaned Artifacts & Checkpoints:
                </div>
                <div className="max-h-36 overflow-y-auto space-y-1 pr-1 font-mono text-[11px]">
                  {housekeepReport.cache_cleanup.cleaned.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between py-0.5 border-b border-slate-800/40 text-slate-400">
                      <span className="truncate max-w-sm text-slate-300">{item.file}</span>
                      <span>{Math.round(item.size_bytes / 1024)} KB</span>
                      <span className="text-slate-500">{item.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Footer with last cycle timestamp */}
            <div className="text-[11px] text-slate-500 flex items-center justify-between pt-1">
              <span>Last Housekeeper cycle: {housekeepReport.timestamp || 'Just now'}</span>
              <span className="text-slate-400 font-mono">Interval: 300s (Background daemon)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

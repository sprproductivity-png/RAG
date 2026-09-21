import React from 'react';
import { Layers, Activity, Cpu, ExternalLink } from 'lucide-react';

export default function Navbar({ health }) {
  const isQdrantConnected = health?.qdrant_connected;
  const qdrantMode = health?.qdrant_mode === 'docker_container' ? 'Docker Container' : 'Local Embedded';
  const isGpu = health?.embedding_device?.toLowerCase().includes('cuda');

  return (
    <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white flex items-center">
              Hybrid RAG & Re-ranking Engine
              <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded-full">
                React + FastAPI
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              Dense + BM25 &bull; Semantic Chunking &bull; Anthropic Contextual Retrieval &bull; Cross-Encoder
            </p>
          </div>
        </div>

        {/* Runtime Status Indicators */}
        <div className="hidden md:flex items-center space-x-2.5">
          {/* Qdrant Status */}
          <div className={`flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${
            isQdrantConnected 
              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/50' 
              : 'bg-rose-950/60 text-rose-300 border-rose-800/50'
          }`}>
            <span className={`w-2 h-2 rounded-full mr-1.5 ${isQdrantConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`}></span>
            Qdrant: {health ? qdrantMode : 'Connecting...'}
          </div>

          {/* Hardware Device */}
          <div className="flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
            <Cpu className="w-3.5 h-3.5 mr-1 text-slate-400" />
            <span>{health ? (health.embedding_device || 'CPU').toUpperCase() : 'AUTO'}</span>
          </div>

          {/* Loaded Models */}
          <div className="text-[11px] text-slate-400 hidden lg:flex items-center space-x-1 bg-slate-900 px-2.5 py-1 rounded border border-slate-800 font-mono">
            <span>Emb: <strong className="text-indigo-300">{health ? health.embedding_model?.split('/').pop() : 'bge-small'}</strong></span>
            <span className="text-slate-600">|</span>
            <span>Rerank: <strong className="text-purple-300">{health ? health.cross_encoder_model?.split('/').pop() : 'ms-marco'}</strong></span>
          </div>

          {/* OpenAPI link */}
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition flex items-center space-x-1"
          >
            <span>Swagger</span>
            <ExternalLink className="w-3 h-3 ml-1 text-slate-400" />
          </a>
        </div>
      </div>
    </header>
  );
}


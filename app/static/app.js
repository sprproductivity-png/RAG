// Self-Hosted Hybrid RAG & Re-ranking Engine - Frontend Logic

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

let currentTab = 'search';
let collectionsList = [];

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initSliders();
    await loadHealth();
    await loadCollections();

    // Default sample search query
    document.getElementById('searchQuery').value = "How long do the batteries last and what is the recharge time?";
});

// Tab Navigation
function initTabs() {
    const tabs = ['search', 'ingest', 'rerank', 'collections'];
    tabs.forEach(tab => {
        const btn = document.getElementById(`tab-btn-${tab}`);
        if (btn) {
            btn.addEventListener('click', () => switchTab(tab));
        }
    });
}

function switchTab(tabName) {
    currentTab = tabName;
    const tabs = ['search', 'ingest', 'rerank', 'collections'];
    tabs.forEach(tab => {
        const view = document.getElementById(`view-${tab}`);
        const btn = document.getElementById(`tab-btn-${tab}`);
        if (tab === tabName) {
            view.classList.remove('hidden');
            btn.classList.add('bg-indigo-600', 'text-white');
            btn.classList.remove('text-slate-400', 'hover:text-slate-200');
        } else {
            view.classList.add('hidden');
            btn.classList.remove('bg-indigo-600', 'text-white');
            btn.classList.add('text-slate-400', 'hover:text-slate-200');
        }
    });

    if (tabName === 'collections') {
        loadCollections();
    }
}

// Sliders and Value Display
function initSliders() {
    const denseSlider = document.getElementById('denseWeightSlider');
    const sparseSlider = document.getElementById('sparseWeightSlider');

    denseSlider.addEventListener('input', (e) => {
        document.getElementById('denseWeightVal').innerText = parseFloat(e.target.value).toFixed(2);
    });

    sparseSlider.addEventListener('input', (e) => {
        document.getElementById('sparseWeightVal').innerText = parseFloat(e.target.value).toFixed(2);
    });
}

// Load System Health & Status Badges
async function loadHealth() {
    try {
        const res = await fetch('/api/v1/health');
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();

        // Qdrant Badge
        const qdrantBadge = document.getElementById('status-qdrant');
        if (data.qdrant_connected) {
            const modeText = data.qdrant_mode === 'docker_container' ? 'Docker Container' : 'Local Embedded';
            qdrantBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span> Qdrant: ${modeText}`;
            qdrantBadge.className = "flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-300 border border-emerald-800/40";
        } else {
            qdrantBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400 mr-1.5"></span> Qdrant: Disconnected`;
            qdrantBadge.className = "flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-rose-950/60 text-rose-300 border border-rose-800/40";
        }

        // Hardware Device Badge
        const deviceBadge = document.getElementById('status-device');
        const isGpu = data.embedding_device.includes('cuda');
        deviceBadge.innerHTML = `<span class="w-2 h-2 rounded-full ${isGpu ? 'bg-indigo-400' : 'bg-amber-400'} mr-1.5"></span> Device: ${data.embedding_device.toUpperCase()}`;

        // Model Badges
        document.getElementById('status-emb-model').innerText = data.embedding_model.split('/').pop();
        document.getElementById('status-rerank-model').innerText = data.cross_encoder_model.split('/').pop();

    } catch (err) {
        console.error("Health check error:", err);
    }
}

// Load Collections List
async function loadCollections() {
    try {
        const res = await fetch('/api/v1/collections');
        if (!res.ok) throw new Error("Failed to load collections");
        const data = await res.json();
        collectionsList = data.collections || [];

        // Update Search Dropdown
        const searchSelect = document.getElementById('searchCollectionSelect');
        searchSelect.innerHTML = "";

        if (collectionsList.length === 0) {
            searchSelect.innerHTML = `<option value="documents">documents (default)</option>`;
        } else {
            collectionsList.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.collection_name;
                opt.innerText = `${c.collection_name} (${c.points_count} chunks)`;
                searchSelect.appendChild(opt);
            });
        }

        // Render Collections Table View
        renderCollectionsTable(collectionsList);

    } catch (err) {
        console.error("Error loading collections:", err);
    }
}

function renderCollectionsTable(collections) {
    const tbody = document.getElementById('collectionsTableBody');
    if (!tbody) return;

    if (collections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center py-8 text-slate-400">No collections found. Ingest a document to create one!</td></tr>`;
        return;
    }

    tbody.innerHTML = collections.map(c => `
        <tr class="border-b border-slate-800/80 hover:bg-slate-800/30">
            <td class="py-3 px-4 font-medium text-indigo-300">${c.collection_name}</td>
            <td class="py-3 px-4 text-slate-300">${c.points_count}</td>
            <td class="py-3 px-4 text-slate-300">${c.vector_size}</td>
            <td class="py-3 px-4 text-slate-300">${c.distance}</td>
            <td class="py-3 px-4 text-slate-300">${c.bm25_docs_count}</td>
            <td class="py-3 px-4 text-right">
                <button onclick="deleteCollection('${c.collection_name}')" class="text-rose-400 hover:text-rose-300 text-xs px-2.5 py-1 rounded bg-rose-950/40 border border-rose-800/40 hover:bg-rose-900/50 transition">
                    Delete
                </button>
            </td>
        </tr>
    `).join('');
}

// Delete Collection
async function deleteCollection(name) {
    if (!confirm(`Are you sure you want to delete collection '${name}' and its BM25 index?`)) {
        return;
    }
    try {
        const res = await fetch(`/api/v1/collections/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error("Delete failed");
        showToast(`Collection '${name}' deleted successfully.`);
        await loadCollections();
    } catch (err) {
        showToast(`Failed to delete collection: ${err.message}`, true);
    }
}

// Execute Hybrid Search
async function executeSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    if (!query) {
        showToast("Please enter a search query.", true);
        return;
    }

    const collectionName = document.getElementById('searchCollectionSelect').value || "documents";
    const topK = parseInt(document.getElementById('topKSelect').value) || 5;
    const denseTopK = parseInt(document.getElementById('denseTopKInput').value) || 15;
    const sparseTopK = parseInt(document.getElementById('sparseTopKInput').value) || 15;
    const rrfK = parseInt(document.getElementById('rrfKInput').value) || 60;
    const denseWeight = parseFloat(document.getElementById('denseWeightSlider').value);
    const sparseWeight = parseFloat(document.getElementById('sparseWeightSlider').value);
    const fusionMethod = document.getElementById('fusionMethodSelect').value;
    const applyReranking = document.getElementById('rerankingToggle').checked;

    const btn = document.getElementById('searchBtn');
    const originalBtnText = btn.innerHTML;
    btn.innerHTML = `<div class="spinner mr-2"></div> Searching...`;
    btn.disabled = true;

    try {
        const res = await fetch('/api/v1/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                collection_name: collectionName,
                top_k: topK,
                dense_top_k: denseTopK,
                sparse_top_k: sparseTopK,
                rrf_k: rrfK,
                dense_weight: denseWeight,
                sparse_weight: sparseWeight,
                fusion_method: fusionMethod,
                apply_reranking: applyReranking
            })
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Search request failed");
        }

        const data = await res.json();
        renderSearchResults(data);

    } catch (err) {
        showToast(`Search error: ${err.message}`, true);
    } finally {
        btn.innerHTML = originalBtnText;
        btn.disabled = false;
    }
}

function renderSearchResults(data) {
    const container = document.getElementById('searchResultsContainer');
    const timings = data.timing_ms || {};

    // Render Latency Bar
    const timingHtml = `
        <div class="glass-panel rounded-xl p-3.5 mb-6 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
            <div class="flex items-center space-x-2">
                <span class="font-semibold text-slate-100">Total Latency:</span>
                <span class="px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 font-mono font-bold">${timings.total_ms || 0} ms</span>
            </div>
            <div class="flex flex-wrap items-center gap-4 text-slate-400">
                <span>Dense: <strong class="text-sky-400 font-mono">${timings.dense_retrieval_ms || 0}ms</strong></span>
                <span>BM25: <strong class="text-amber-400 font-mono">${timings.sparse_retrieval_ms || 0}ms</strong></span>
                <span>Fusion: <strong class="text-emerald-400 font-mono">${timings.fusion_ms || 0}ms</strong></span>
                <span>Re-ranker: <strong class="text-purple-400 font-mono">${timings.reranking_ms || 0}ms</strong></span>
            </div>
            <div class="text-slate-400">
                Candidates: ${data.total_fused_candidates} fused (${data.total_dense_candidates} dense, ${data.total_sparse_candidates} sparse)
            </div>
        </div>
    `;

    if (!data.results || data.results.length === 0) {
        container.innerHTML = timingHtml + `
            <div class="text-center py-12 glass-panel rounded-2xl">
                <p class="text-slate-400 text-base">No matching documents found in this collection.</p>
                <p class="text-slate-500 text-xs mt-1">Try ingesting sample documents in the Ingestion Studio tab first.</p>
            </div>
        `;
        return;
    }

    const cardsHtml = data.results.map((r, i) => {
        const ceScore = r.cross_encoder_score !== null ? r.cross_encoder_score.toFixed(4) : "N/A";
        const denseScore = r.dense_score !== null ? r.dense_score.toFixed(4) : "N/A";
        const bm25Score = r.bm25_score !== null ? r.bm25_score.toFixed(4) : "N/A";
        const fusionScore = r.fusion_score !== null ? r.fusion_score.toFixed(6) : "N/A";

        return `
            <div class="glass-panel card-hover rounded-xl p-5 mb-4 border border-slate-800 relative animate-fade-in">
                <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
                    <div class="flex items-center space-x-2.5">
                        <span class="flex items-center justify-center w-7 h-7 rounded-lg bg-indigo-600/30 text-indigo-300 font-bold text-sm border border-indigo-500/30">
                            #${i + 1}
                        </span>
                        <span class="text-xs text-slate-400 font-mono">${r.chunk_id.slice(0, 8)}...</span>
                        ${r.metadata && r.metadata.title ? `<span class="text-xs font-medium text-slate-300 bg-slate-800 px-2 py-0.5 rounded">${r.metadata.title}</span>` : ''}
                    </div>

                    <!-- Score Badges -->
                    <div class="flex flex-wrap items-center gap-2">
                        ${r.cross_encoder_score !== null ? `
                            <div class="badge-cross-encoder px-2.5 py-1 rounded-md text-xs font-semibold flex items-center shadow-sm">
                                <span>Cross-Encoder: <strong>${ceScore}</strong></span>
                            </div>
                        ` : ''}
                        <div class="badge-dense px-2 py-0.5 rounded text-xs">
                            Dense: <strong>${denseScore}</strong> ${r.dense_rank ? `(#${r.dense_rank})` : ''}
                        </div>
                        <div class="badge-bm25 px-2 py-0.5 rounded text-xs">
                            BM25: <strong>${bm25Score}</strong> ${r.bm25_rank ? `(#${r.bm25_rank})` : ''}
                        </div>
                        <div class="badge-rrf px-2 py-0.5 rounded text-xs">
                            RRF: <strong>${fusionScore}</strong>
                        </div>
                    </div>
                </div>

                <!-- Anthropic Contextual Retrieval Summary Box -->
                ${r.context_summary ? `
                    <div class="badge-context rounded-lg p-3 mb-3 text-xs leading-relaxed">
                        <div class="flex items-center font-semibold mb-1 text-purple-300">
                            <svg class="w-3.5 h-3.5 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            Anthropic Contextual Retrieval Summary
                        </div>
                        <span class="text-purple-200/90">${r.context_summary}</span>
                    </div>
                ` : ''}

                <!-- Raw Chunk Text -->
                <div class="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap bg-slate-900/40 p-3 rounded-lg border border-slate-800/60 font-normal">
                    ${escapeHtml(r.text)}
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = timingHtml + cardsHtml;
}

// Ingestion Studio
function loadPresetDoc(name) {
    const preset = PRESETS[name];
    if (!preset) return;
    document.getElementById('ingestTitle').value = preset.title;
    document.getElementById('ingestCategory').value = preset.category;
    document.getElementById('ingestText').value = preset.text;
    showToast(`Loaded preset: ${preset.title}`);
}

async function executeIngest() {
    const title = document.getElementById('ingestTitle').value.trim();
    const category = document.getElementById('ingestCategory').value.trim();
    const text = document.getElementById('ingestText').value.trim();
    const collectionName = document.getElementById('ingestCollectionName').value.trim() || "documents";
    const chunkingStrategy = document.getElementById('chunkingStrategySelect').value;
    const useContextual = document.getElementById('contextualToggle').checked;
    const contextProvider = document.getElementById('contextProviderSelect').value;

    if (!text) {
        showToast("Please provide document text to ingest.", true);
        return;
    }

    const btn = document.getElementById('ingestBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<div class="spinner mr-2"></div> Ingesting & Contextualizing...`;
    btn.disabled = true;

    try {
        const res = await fetch('/api/v1/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                collection_name: collectionName,
                chunking_strategy: chunkingStrategy,
                use_contextual_retrieval: useContextual,
                context_provider: contextProvider,
                documents: [
                    {
                        title: title || "Untitled Document",
                        text: text,
                        metadata: {
                            title: title,
                            category: category,
                            ingested_at: new Date().toISOString()
                        }
                    }
                ]
            })
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Ingestion failed");
        }

        const data = await res.json();
        showToast(`Ingestion complete: ${data.chunks_created} chunks indexed into '${data.collection_name}' (${data.duration_ms}ms)`);

        // Render Sample Chunks Preview
        renderIngestPreview(data);
        await loadCollections();

    } catch (err) {
        showToast(`Ingestion error: ${err.message}`, true);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function renderIngestPreview(data) {
    const container = document.getElementById('ingestPreviewContainer');
    if (!container || !data.sample_chunks) return;

    container.classList.remove('hidden');
    const list = document.getElementById('ingestChunksList');
    list.innerHTML = data.sample_chunks.map((c, i) => `
        <div class="glass-panel p-4 rounded-xl border border-slate-800 mb-3 text-xs">
            <div class="flex items-center justify-between mb-2">
                <span class="font-bold text-indigo-400">Chunk #${c.chunk_index}</span>
                <span class="text-slate-400 font-mono">${c.chunk_id.slice(0, 8)}...</span>
            </div>
            ${c.context_summary ? `
                <div class="badge-context p-2.5 rounded-lg mb-2 text-purple-200">
                    <strong>Contextual Summary:</strong> ${c.context_summary}
                </div>
            ` : ''}
            <div class="text-slate-300 bg-slate-900/50 p-2.5 rounded border border-slate-800">
                ${escapeHtml(c.raw_text)}
            </div>
        </div>
    `).join('');
}

// Standalone Cross-Encoder Playground
async function executeRerank() {
    const query = document.getElementById('rerankQuery').value.trim();
    const p1 = document.getElementById('passage1').value.trim();
    const p2 = document.getElementById('passage2').value.trim();
    const p3 = document.getElementById('passage3').value.trim();

    if (!query) {
        showToast("Please enter a query to test re-ranking.", true);
        return;
    }

    const passages = [p1, p2, p3].filter(p => p.length > 0);
    if (passages.length === 0) {
        showToast("Please enter at least one passage.", true);
        return;
    }

    const btn = document.getElementById('rerankBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<div class="spinner mr-2"></div> Scoring...`;
    btn.disabled = true;

    try {
        const res = await fetch('/api/v1/rerank', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                documents: passages.map((text, idx) => ({ id: `cand_${idx + 1}`, text }))
            })
        });

        if (!res.ok) throw new Error("Re-ranking failed");
        const data = await res.json();

        // Render re-ranked results
        const container = document.getElementById('rerankResultsContainer');
        container.innerHTML = data.results.map((r, rank) => `
            <div class="glass-panel card-hover p-4 rounded-xl border border-slate-800 mb-3 animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <span class="flex items-center justify-center w-6 h-6 rounded bg-purple-600/30 text-purple-300 font-bold text-xs">
                        #${rank + 1}
                    </span>
                    <span class="badge-cross-encoder px-2.5 py-1 rounded text-xs font-semibold">
                        Cross-Encoder Logit: ${r.score}
                    </span>
                </div>
                <div class="text-slate-200 text-xs leading-relaxed bg-slate-900/40 p-3 rounded">
                    ${escapeHtml(r.text)}
                </div>
            </div>
        `).join('');

    } catch (err) {
        showToast(`Re-ranking error: ${err.message}`, true);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// Utility: Toast Notification
function showToast(message, isError = false) {
    const toast = document.getElementById('toastNotification');
    toast.innerText = message;
    toast.className = `fixed bottom-6 right-6 px-4 py-3 rounded-xl text-sm font-medium shadow-2xl z-50 transition-all duration-300 ${
        isError ? 'bg-rose-600 text-white' : 'bg-emerald-600 text-white'
    }`;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 4000);
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
}


// API Client for FastAPI Hybrid RAG Backend

const API_BASE = '/api/v1';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch system health');
  return res.json();
}

export async function fetchCollections() {
  const res = await fetch(`${API_BASE}/collections`);
  if (!res.ok) throw new Error('Failed to fetch collections');
  return res.json();
}

export async function deleteCollectionApi(name) {
  const res = await fetch(`${API_BASE}/collections/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to delete collection');
  }
  return res.json();
}

export async function executeSearchApi(params) {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Search request failed');
  }
  return res.json();
}

export async function executeIngestApi(payload) {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Ingestion request failed');
  }
  return res.json();
}

export async function executeRerankApi(payload) {
  const res = await fetch(`${API_BASE}/rerank`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Re-ranking request failed');
  }
  return res.json();
}


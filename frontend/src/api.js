const BASE = '/api';

export async function fetchSummary() {
    const res = await fetch(`${BASE}/summary`);
    if (!res.ok) throw new Error(`summary: ${res.status}`);
    return res.json();
}

export async function fetchGraphStats() {
    const res = await fetch(`${BASE}/graph-stats`);
    if (!res.ok) throw new Error(`graph-stats: ${res.status}`);
    return res.json();
}

export async function fetchMetrics() {
    const res = await fetch(`${BASE}/metrics`);
    if (!res.ok) throw new Error(`metrics: ${res.status}`);
    return res.json();
}

export async function fetchAccounts({ category, sortBy, order, page, limit, search } = {}) {
    const params = new URLSearchParams();
    if (category) params.set('category', category);
    if (sortBy) params.set('sort_by', sortBy);
    if (order) params.set('order', order);
    if (page) params.set('page', String(page));
    if (limit) params.set('limit', String(limit));
    if (search) params.set('search', search);
    const res = await fetch(`${BASE}/accounts?${params}`);
    if (!res.ok) throw new Error(`accounts: ${res.status}`);
    return res.json();
}

export async function fetchAccount(accountId) {
    const res = await fetch(`${BASE}/accounts/${encodeURIComponent(accountId)}`);
    if (!res.ok) throw new Error(`account ${accountId}: ${res.status}`);
    return res.json();
}

export async function predictTransaction(txData) {
    const res = await fetch(`${BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: [txData] }),
    });
    if (!res.ok) throw new Error(`predict: ${res.status}`);
    return res.json();
}

export async function fetchHealth() {
    const res = await fetch(`${BASE}/health`);
    if (!res.ok) throw new Error(`health: ${res.status}`);
    return res.json();
}

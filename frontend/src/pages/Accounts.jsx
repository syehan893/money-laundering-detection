import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { fetchAccounts } from '../api'

function formatMoney(n) {
    if (n == null) return '—'
    if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
    return `$${n.toFixed(2)}`
}

function getRiskClass(cat) {
    return (cat || '').toLowerCase()
}

const CATEGORIES = ['Low', 'Moderate', 'High', 'Critical']
const DOT_CLASSES = { Low: 'low', Moderate: 'moderate', High: 'high', Critical: 'critical' }

export default function Accounts() {
    const navigate = useNavigate()
    const [searchParams, setSearchParams] = useSearchParams()

    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    const activeCategory = searchParams.get('category') || ''
    const currentPage = parseInt(searchParams.get('page') || '1', 10)
    const searchQuery = searchParams.get('search') || ''
    const [searchInput, setSearchInput] = useState(searchQuery)

    const load = useCallback(async () => {
        setLoading(true)
        setError(null)
        try {
            const result = await fetchAccounts({
                category: activeCategory || undefined,
                page: currentPage,
                limit: 20,
                sortBy: 'risk_score',
                order: 'desc',
                search: searchQuery || undefined,
            })
            setData(result)
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }, [activeCategory, currentPage, searchQuery])

    useEffect(() => { load() }, [load])

    const setCategory = (cat) => {
        const params = new URLSearchParams(searchParams)
        if (cat) params.set('category', cat); else params.delete('category')
        params.set('page', '1')
        setSearchParams(params)
    }

    const setPage = (p) => {
        const params = new URLSearchParams(searchParams)
        params.set('page', String(p))
        setSearchParams(params)
    }

    const handleSearch = (e) => {
        e.preventDefault()
        const params = new URLSearchParams(searchParams)
        if (searchInput) params.set('search', searchInput); else params.delete('search')
        params.set('page', '1')
        setSearchParams(params)
    }

    const pagination = data?.pagination || {}
    const accounts = data?.accounts || []
    const catCounts = data?.category_counts || {}

    const totalPages = pagination.total_pages || 1
    const pageNumbers = []
    for (let i = 1; i <= Math.min(totalPages, 9); i++) pageNumbers.push(i)
    if (totalPages > 9) {
        const pages = new Set([1, 2, 3, currentPage - 1, currentPage, currentPage + 1, totalPages - 1, totalPages])
        const sorted = [...pages].filter(p => p >= 1 && p <= totalPages).sort((a, b) => a - b)
        pageNumbers.length = 0
        sorted.forEach((p, i) => {
            if (i > 0 && p - sorted[i - 1] > 1) pageNumbers.push('...')
            pageNumbers.push(p)
        })
    }

    return (
        <div className="fade-in">
            <div className="page-header">
                <h2>Accounts Overview</h2>
                <p>Monitor and manage high-risk financial entities and their activities.</p>
            </div>

            {/* Filter Bar */}
            <div className="filter-bar">
                <button
                    className={`filter-tab ${!activeCategory ? 'active' : ''}`}
                    onClick={() => setCategory('')}
                >
                    All
                </button>
                {CATEGORIES.map(cat => (
                    <button
                        key={cat}
                        className={`filter-tab ${activeCategory === cat ? 'active' : ''}`}
                        onClick={() => setCategory(cat)}
                    >
                        <span className={`dot ${DOT_CLASSES[cat]}`} />
                        {cat} ({catCounts[cat] != null ? catCounts[cat].toLocaleString() : '—'})
                    </button>
                ))}

                <form onSubmit={handleSearch} className="search-wrapper">
                    <span className="material-symbols-outlined">search</span>
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search by Account ID..."
                        value={searchInput}
                        onChange={e => setSearchInput(e.target.value)}
                    />
                </form>
            </div>

            {/* Table */}
            {loading ? (
                <div className="loading-container"><div className="spinner" /><p>Loading accounts...</p></div>
            ) : error ? (
                <div className="error-container">
                    <span className="material-symbols-outlined" style={{ fontSize: 48 }}>error</span>
                    <p>{error}</p>
                </div>
            ) : (
                <div className="glass-card" style={{ overflow: 'hidden' }}>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Account ID</th>
                                    <th>Risk Score</th>
                                    <th>Risk Category</th>
                                    <th>Total Sent ($)</th>
                                    <th>Total Received ($)</th>
                                    <th>Partners</th>
                                    <th style={{ textAlign: 'right' }}>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {accounts.map((acc, i) => {
                                    const rc = getRiskClass(acc.risk_category)
                                    const score = acc.risk_score != null ? Number(acc.risk_score).toFixed(0) : 0
                                    return (
                                        <tr key={acc.account_id || i}>
                                            <td style={{ fontWeight: 500, color: 'var(--text-white)' }}>{acc.account_id}</td>
                                            <td>
                                                <div className="score-bar-track">
                                                    <div className={`score-bar-fill ${rc}`} style={{ width: `${score}%` }} />
                                                </div>
                                                <span className="score-text">{score}/100</span>
                                            </td>
                                            <td><span className={`risk-badge ${rc}`}>{acc.risk_category}</span></td>
                                            <td className="mono">{formatMoney(acc.total_sent)}</td>
                                            <td className="mono">{formatMoney(acc.total_received)}</td>
                                            <td style={{ color: 'var(--text-secondary)' }}>{acc.unique_partners ?? '—'}</td>
                                            <td style={{ textAlign: 'right' }}>
                                                <button className="btn-review" onClick={() => navigate(`/accounts/${acc.account_id}`)}>
                                                    View
                                                </button>
                                            </td>
                                        </tr>
                                    )
                                })}
                                {accounts.length === 0 && (
                                    <tr><td colSpan={7} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>No accounts found</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>

                    {/* Pagination */}
                    <div className="pagination">
                        <span className="pagination-info">
                            Showing <strong>{pagination.from || 1}</strong> to <strong>{pagination.to || accounts.length}</strong> of <strong>{(pagination.total || 0).toLocaleString()}</strong> results
                        </span>
                        <div className="pagination-buttons">
                            <button className="page-btn" disabled={currentPage <= 1} onClick={() => setPage(currentPage - 1)}>‹</button>
                            {pageNumbers.map((p, i) =>
                                p === '...'
                                    ? <span key={`e${i}`} style={{ padding: '0 6px', color: 'var(--text-muted)' }}>…</span>
                                    : <button key={p} className={`page-btn ${p === currentPage ? 'active' : ''}`} onClick={() => setPage(p)}>{p}</button>
                            )}
                            <button className="page-btn" disabled={currentPage >= totalPages} onClick={() => setPage(currentPage + 1)}>›</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

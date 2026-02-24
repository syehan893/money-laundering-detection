import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { fetchSummary, fetchGraphStats } from '../api'

const RISK_COLORS = {
    Low: '#10b981',
    Moderate: '#f59e0b',
    High: '#ef4444',
    Critical: '#7f1d1d',
}

const CURRENCY_COLORS = ['#3b82f6', '#6366f1', '#06b6d4', '#14b8a6', '#f97316']

function formatNumber(n) {
    if (n == null) return '—'
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
    return n.toLocaleString()
}

function formatMoney(n) {
    if (n == null) return '—'
    if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
    return `$${n.toFixed(0)}`
}

function getRiskClass(cat) {
    return (cat || '').toLowerCase()
}

export default function Dashboard() {
    const [summary, setSummary] = useState(null)
    const [graphStats, setGraphStats] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const navigate = useNavigate()

    useEffect(() => {
        Promise.all([fetchSummary(), fetchGraphStats()])
            .then(([s, g]) => { setSummary(s); setGraphStats(g) })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }, [])

    if (loading) return (
        <div className="loading-container">
            <div className="spinner" />
            <p>Loading dashboard...</p>
        </div>
    )

    if (error) return (
        <div className="error-container">
            <span className="material-symbols-outlined" style={{ fontSize: 48 }}>error</span>
            <p>{error}</p>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>Make sure the API is running on port 8000</p>
        </div>
    )

    const overview = summary?.overview || {}
    const riskDist = summary?.risk_distribution?.accounts || {}
    const txRiskDist = summary?.risk_distribution?.transactions || {}
    const topFlagged = summary?.top_flagged_accounts || []
    const currencyStats = summary?.currency_stats || []
    const modelMetrics = summary?.model_metrics || {}

    // Donut chart data
    const donutData = Object.entries(riskDist)
        .filter(([, v]) => v > 0)
        .map(([name, value]) => ({ name, value }))

    const totalRisk = donutData.reduce((s, d) => s + d.value, 0)

    // Currency bars data - top 5
    const topCurrencies = [...currencyStats]
        .sort((a, b) => (b.total_amount || 0) - (a.total_amount || 0))
        .slice(0, 5)
    const maxCurrVol = topCurrencies[0]?.total_amount || 1

    return (
        <div className="fade-in">
            {/* Header */}
            <div className="page-header">
                <h2>Overview</h2>
                <p>Real-time monitoring and risk assessment.</p>
            </div>

            {/* KPI Grid */}
            <div className="kpi-grid">
                <div className="glass-card kpi-card">
                    <div className="kpi-icon"><span className="material-symbols-outlined" style={{ color: '#3b82f6' }}>account_balance</span></div>
                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <p className="kpi-label">Total Accounts</p>
                        <p className="kpi-value">{formatNumber(overview.total_accounts)}</p>
                        <span className="kpi-badge success">
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>check_circle</span>
                            Active
                        </span>
                    </div>
                </div>

                <div className="glass-card kpi-card">
                    <div className="kpi-icon"><span className="material-symbols-outlined" style={{ color: '#a855f7' }}>payments</span></div>
                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <p className="kpi-label">Transactions</p>
                        <p className="kpi-value">{formatNumber(overview.total_transactions)}</p>
                        <span className="kpi-badge success">
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>trending_up</span>
                            Processed
                        </span>
                    </div>
                </div>

                <div className="glass-card kpi-card flagged">
                    <div className="kpi-icon"><span className="material-symbols-outlined" style={{ color: '#ef4444' }}>flag</span></div>
                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <p className="kpi-label">Flagged</p>
                        <p className="kpi-value">{formatNumber((riskDist.High || 0) + (riskDist.Critical || 0))}</p>
                        <span className="kpi-badge danger">
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>warning</span>
                            High + Critical
                        </span>
                    </div>
                </div>

                <div className="glass-card kpi-card">
                    <div className="kpi-icon"><span className="material-symbols-outlined" style={{ color: '#10b981' }}>analytics</span></div>
                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <p className="kpi-label">Model F1 Score</p>
                        <p className="kpi-value">{modelMetrics.test_metrics?.f1 != null ? modelMetrics.test_metrics.f1.toFixed(3) : '—'}</p>
                        <span className="kpi-badge success">
                            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>check_circle</span>
                            Stable
                        </span>
                    </div>
                </div>
            </div>

            {/* Charts Section */}
            <div className="charts-section">
                {/* Risk Distribution Donut */}
                <div className="glass-card chart-card">
                    <h3>Risk Distribution</h3>
                    <div style={{ width: '100%', height: 220, position: 'relative' }}>
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie
                                    data={donutData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={90}
                                    dataKey="value"
                                    stroke="none"
                                >
                                    {donutData.map((entry) => (
                                        <Cell key={entry.name} fill={RISK_COLORS[entry.name] || '#666'} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#e2e8f0' }}
                                    formatter={(value, name) => [`${formatNumber(value)} (${((value / totalRisk) * 100).toFixed(1)}%)`, name]}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                        {/* Center label */}
                        <div style={{
                            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                            textAlign: 'center', pointerEvents: 'none'
                        }}>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Total</div>
                            <div style={{ fontSize: 20, fontWeight: 700, color: 'white' }}>{formatNumber(totalRisk)}</div>
                        </div>
                    </div>
                    <div className="donut-legend">
                        {Object.entries(RISK_COLORS).map(([name, color]) => (
                            <div key={name} className="donut-legend-item">
                                <div className="donut-legend-dot" style={{ background: color }} />
                                <span>{name} ({riskDist[name] != null ? formatNumber(riskDist[name]) : 0})</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Transactions by Currency */}
                <div className="glass-card chart-card">
                    <h3>Transactions by Currency</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, justifyContent: 'center', flex: 1 }}>
                        {topCurrencies.map((c, i) => (
                            <div key={c.currency || i} className="currency-bar">
                                <div className="currency-bar-header">
                                    <span className="name">{c.currency}</span>
                                    <span className="value">{formatMoney(c.total_amount)}</span>
                                </div>
                                <div className="currency-bar-track">
                                    <div
                                        className="currency-bar-fill"
                                        style={{
                                            width: `${((c.total_amount || 0) / maxCurrVol) * 100}%`,
                                            background: CURRENCY_COLORS[i % CURRENCY_COLORS.length],
                                        }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Top Flagged Accounts */}
            <div className="glass-card" style={{ overflow: 'hidden' }}>
                <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 className="section-title" style={{ margin: 0 }}>Top Flagged Accounts</h3>
                </div>
                <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Account ID</th>
                                <th>Risk Score</th>
                                <th>Category</th>
                                <th>Total Sent</th>
                                <th>Total Received</th>
                                <th style={{ textAlign: 'right' }}>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {topFlagged.slice(0, 10).map((acc, i) => {
                                const riskClass = getRiskClass(acc.risk_category)
                                const score = (acc.risk_score != null ? Number(acc.risk_score).toFixed(0) : 0)
                                return (
                                    <tr key={acc.account_id || i}>
                                        <td style={{ fontWeight: 500, color: 'var(--text-white)' }}>{acc.account_id}</td>
                                        <td>
                                            <div className="score-bar-track">
                                                <div className={`score-bar-fill ${riskClass}`} style={{ width: `${score}%` }} />
                                            </div>
                                            <span className="score-text">{score}/100</span>
                                        </td>
                                        <td><span className={`risk-badge ${riskClass}`}>{acc.risk_category}</span></td>
                                        <td className="mono">{formatMoney(acc.total_sent)}</td>
                                        <td className="mono">{formatMoney(acc.total_received)}</td>
                                        <td style={{ textAlign: 'right' }}>
                                            <button className="btn-review" onClick={() => navigate(`/accounts/${acc.account_id}`)}>
                                                Review
                                            </button>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

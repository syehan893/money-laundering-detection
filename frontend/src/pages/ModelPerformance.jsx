import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { fetchMetrics } from '../api'

function formatPct(val) {
    if (val == null) return '—'
    return `${(val * 100).toFixed(2)}%`
}

export default function ModelPerformance() {
    const [metrics, setMetrics] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        fetchMetrics()
            .then(d => setMetrics(d))
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }, [])

    if (loading) return <div className="loading-container"><div className="spinner" /><p>Loading metrics...</p></div>
    if (error) return <div className="error-container"><span className="material-symbols-outlined" style={{ fontSize: 48 }}>error</span><p>{error}</p></div>
    if (!metrics) return null

    const perf = metrics.model_performance || {}
    const test = perf.test_metrics || {}
    const cm = perf.confusion_matrix || {}
    const hp = metrics.hyperparameters || {}
    const history = (metrics.training_history || []).map((h, i) => ({
        epoch: h.epoch || i + 1,
        train_loss: h.train_loss,
        val_f1: h.val_f1,
    }))

    const accuracy = cm.tn != null && cm.fp != null && cm.fn != null && cm.tp != null
        ? (cm.tn + cm.tp) / (cm.tn + cm.fp + cm.fn + cm.tp)
        : null

    const metricCards = [
        { label: 'Precision', value: test.precision, color: '#10b981' },
        { label: 'Recall (Sensitivity)', value: test.recall, color: '#f59e0b' },
        { label: 'F1 Score', value: test.f1, color: '#3b82f6', raw: true },
        { label: 'Accuracy', value: accuracy, color: '#a855f7' },
    ]

    const hyperparams = [
        { param: 'learning_rate', value: hp.lr, desc: 'Step size for gradient descent', impact: 'High' },
        { param: 'hidden_dim', value: hp.hidden_dim, desc: 'Size of hidden layers in Transformer', impact: 'Medium' },
        { param: 'attention_heads', value: hp.num_heads, desc: 'Number of heads in Multi-Head Attention', impact: 'Medium' },
        { param: 'focal_alpha', value: hp.focal_alpha, desc: 'Class weight for Focal Loss', impact: 'High' },
        { param: 'focal_gamma', value: hp.focal_gamma, desc: 'Focusing parameter for Focal Loss', impact: 'Medium' },
        { param: 'oversample_ratio', value: hp.oversample_ratio, desc: 'Negative-to-positive sampling ratio', impact: 'High' },
        { param: 'dropout', value: hp.dropout, desc: 'Regularization dropout rate', impact: 'Low' },
        { param: 'threshold', value: hp.threshold, desc: 'Optimal classification threshold', impact: 'Critical' },
    ]

    return (
        <div className="fade-in">
            <div className="page-header">
                <h2>AML Detection Model</h2>
                <p>Performance analytics and training insights</p>
            </div>

            {/* Metric Cards */}
            <div className="metrics-grid">
                {metricCards.map(m => (
                    <div key={m.label} className="glass-card metric-card">
                        <div className="metric-label">{m.label}</div>
                        <div className="metric-value">
                            {m.raw ? (m.value?.toFixed(4) || '—') : formatPct(m.value)}
                        </div>
                        <div className="metric-bar">
                            <div className="metric-bar-fill" style={{ width: `${(m.value || 0) * 100}%`, background: m.color }} />
                        </div>
                    </div>
                ))}
            </div>

            {/* Charts Row */}
            <div className="charts-row">
                {/* Training History */}
                <div className="glass-card chart-card">
                    <h3>Training History</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: -16, marginBottom: 24 }}>
                        Loss vs Validation F1 ({history.length} Epochs)
                    </p>
                    <div style={{ width: '100%', height: 320 }}>
                        <ResponsiveContainer>
                            <LineChart data={history}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 12 }} />
                                <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#e2e8f0', fontSize: 13 }}
                                />
                                <Legend wrapperStyle={{ fontSize: 13 }} />
                                <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" name="Training Loss" dot={false} strokeWidth={2} connectNulls />
                                <Line type="monotone" dataKey="val_f1" stroke="#10b981" name="Validation F1" dot={false} strokeWidth={2} connectNulls />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Confusion Matrix */}
                <div className="glass-card chart-card">
                    <h3>Confusion Matrix</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
                        <div style={{ textAlign: 'center', fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                            Predicted Class
                        </div>
                        <div className="confusion-grid">
                            <div className="cm-cell tn">
                                <div className="cm-type">True Negative</div>
                                <div className="cm-value">{(cm.tn || 0).toLocaleString()}</div>
                                <div className="cm-desc">Legitimate</div>
                            </div>
                            <div className="cm-cell fp">
                                <div className="cm-type">False Positive</div>
                                <div className="cm-value">{(cm.fp || 0).toLocaleString()}</div>
                                <div className="cm-desc">False Alarm</div>
                            </div>
                            <div className="cm-cell fn">
                                <div className="cm-type">False Negative</div>
                                <div className="cm-value">{(cm.fn || 0).toLocaleString()}</div>
                                <div className="cm-desc">Missed Fraud</div>
                            </div>
                            <div className="cm-cell tp">
                                <div className="cm-type">True Positive</div>
                                <div className="cm-value">{(cm.tp || 0).toLocaleString()}</div>
                                <div className="cm-desc">Caught Fraud</div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
                            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', writingMode: 'horizontal-tb' }}>
                                Actual Class
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Hyperparameters */}
            <div className="glass-card" style={{ overflow: 'hidden' }}>
                <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <h3 className="section-title" style={{ margin: 0 }}>Model Hyperparameters</h3>
                </div>
                <table className="params-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                            <th>Description</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {hyperparams.map(h => (
                            <tr key={h.param}>
                                <td>{h.param}</td>
                                <td>{h.value != null ? h.value : '—'}</td>
                                <td style={{ color: 'var(--text-secondary)' }}>{h.desc}</td>
                                <td>
                                    <span className={`risk-badge ${h.impact === 'Critical' ? 'critical' : h.impact === 'High' ? 'high' : h.impact === 'Medium' ? 'moderate' : 'low'}`}>
                                        {h.impact}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

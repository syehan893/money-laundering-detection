import { useState } from 'react'
import { predictTransaction } from '../api'

function getRiskColor(cat) {
    const c = (cat || '').toLowerCase()
    if (c === 'critical') return '#dc2626'
    if (c === 'high') return '#ef4444'
    if (c === 'moderate') return '#f59e0b'
    return '#10b981'
}

function getRiskClass(cat) {
    return (cat || '').toLowerCase()
}

const CURRENCIES = ['US Dollar', 'Euro', 'UK Pound', 'Swiss Franc', 'Bitcoin', 'Australian Dollar', 'Canadian Dollar', 'Japanese Yen', 'Mexican Peso', 'Brazil Real', 'Yuan', 'Ruble', 'Rupee', 'Saudi Riyal', 'Swedish Krona', 'Shekel', 'Rand']

const LOCATIONS = ['US', 'UK', 'China', 'Germany', 'France', 'Russia', 'Switzerland', 'Japan', 'India', 'Mexico', 'Brazil', 'Australia', 'Canada', 'Italy', 'Spain', 'Saudi Arabia', 'South Africa', 'Sweden', 'Israel', 'Turkey', 'UAE', 'Hong Kong']

const PAYMENT_TYPES = ['Credit Card', 'Wire', 'ACH', 'Cheque', 'Cash', 'Reinvestment', 'Crypto', 'Debit Card']

const INITIAL_FORM = {
    sender_account: '',
    receiver_account: '',
    amount: '',
    payment_currency: 'US Dollar',
    received_currency: 'US Dollar',
    sender_bank_location: 'US',
    receiver_bank_location: 'US',
    payment_type: 'Wire',
    date: '',
    time: '',
}

export default function Predict() {
    const [form, setForm] = useState(INITIAL_FORM)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const update = (field, value) => setForm(f => ({ ...f, [field]: value }))

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const txData = {
                From_Account: form.sender_account,
                To_Account: form.receiver_account,
                Amount_Paid: parseFloat(form.amount) || 0,
                Payment_Currency: form.payment_currency,
                Received_Currency: form.received_currency,
                Sender_Bank_Location: form.sender_bank_location,
                Receiver_Bank_Location: form.receiver_bank_location,
                Payment_Type: form.payment_type,
                Date: form.date || '2023-01-01',
                Time: form.time ? `${form.time}:00` : '12:00:00',
            }
            const res = await predictTransaction(txData)
            if (res.results?.length > 0) {
                setResult(res.results[0])
            }
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }

    const isCrossBorder = form.sender_bank_location !== form.receiver_bank_location
    const isForeignCurrency = form.payment_currency !== form.received_currency
    const probability = result?.probability ?? null
    const riskCategory = result?.risk_category ?? null
    const riskColor = getRiskColor(riskCategory)
    const pctValue = probability != null ? (probability * 100).toFixed(1) : 0

    return (
        <div className="fade-in">
            <div className="page-header">
                <h2>Predict Transaction Risk</h2>
                <p>Analyze potential money laundering activities in real-time.</p>
            </div>

            <div className="predict-layout">
                {/* Form */}
                <div className="glass-card form-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 28 }}>
                        <span className="material-symbols-outlined" style={{ color: 'var(--primary)' }}>receipt_long</span>
                        <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-white)', margin: 0 }}>Transaction Details</h3>
                    </div>

                    <form onSubmit={handleSubmit}>
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">Sender Account</label>
                                <input className="form-input" placeholder="Account Number" value={form.sender_account} onChange={e => update('sender_account', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Receiver Account</label>
                                <input className="form-input" placeholder="Account Number" value={form.receiver_account} onChange={e => update('receiver_account', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Amount</label>
                                <input className="form-input" type="number" step="0.01" placeholder="0.00" value={form.amount} onChange={e => update('amount', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Currency</label>
                                <select className="form-select" value={form.payment_currency} onChange={e => update('payment_currency', e.target.value)}>
                                    {CURRENCIES.map(c => <option key={c}>{c}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Date</label>
                                <input className="form-input" type="date" value={form.date} onChange={e => update('date', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Time</label>
                                <input className="form-input" type="time" value={form.time} onChange={e => update('time', e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Origin Bank Location</label>
                                <select className="form-select" value={form.sender_bank_location} onChange={e => update('sender_bank_location', e.target.value)}>
                                    {LOCATIONS.map(l => <option key={l}>{l}</option>)}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Dest. Bank Location</label>
                                <select className="form-select" value={form.receiver_bank_location} onChange={e => update('receiver_bank_location', e.target.value)}>
                                    {LOCATIONS.map(l => <option key={l}>{l}</option>)}
                                </select>
                            </div>
                            <div className="form-group full-width">
                                <label className="form-label">Payment Type</label>
                                <select className="form-select" value={form.payment_type} onChange={e => update('payment_type', e.target.value)}>
                                    {PAYMENT_TYPES.map(p => <option key={p}>{p}</option>)}
                                </select>
                            </div>
                        </div>

                        <button className="submit-btn" type="submit" disabled={loading}>
                            <span className="material-symbols-outlined">security</span>
                            {loading ? 'Analyzing...' : 'Analyze Transaction'}
                        </button>
                    </form>
                </div>

                {/* Results */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                    <div className="glass-card result-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
                            <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-white)', margin: 0 }}>Risk Assessment</h3>
                            {riskCategory && (
                                <span className={`risk-badge ${getRiskClass(riskCategory)}`} style={{ fontSize: 13, padding: '4px 14px' }}>
                                    {riskCategory?.toUpperCase()} RISK DETECTED
                                </span>
                            )}
                        </div>

                        {result == null && !error ? (
                            <div className="result-empty">
                                <span className="material-symbols-outlined">psychology</span>
                                <p>Submit a transaction to see risk analysis results</p>
                            </div>
                        ) : error ? (
                            <div className="error-container" style={{ minHeight: 200 }}>
                                <span className="material-symbols-outlined" style={{ fontSize: 40 }}>error</span>
                                <p>{error}</p>
                            </div>
                        ) : (
                            <>
                                {/* Gauge */}
                                <div className="risk-gauge-large" style={{
                                    background: `conic-gradient(${riskColor} ${pctValue * 3.6}deg, rgba(255,255,255,0.05) 0deg)`,
                                }}>
                                    <div style={{
                                        width: 160, height: 160, borderRadius: '50%', background: 'var(--bg-primary)',
                                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                                    }}>
                                        <span className="gauge-value">{pctValue}%</span>
                                        <span className="gauge-label" style={{ color: riskColor }}>PROBABILITY</span>
                                    </div>
                                </div>
                                <p style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: 14 }}>
                                    This transaction has a {riskCategory?.toLowerCase()} probability of being linked to money laundering activities.
                                </p>
                            </>
                        )}
                    </div>

                    {/* Risk Factors */}
                    {result && (
                        <div className="glass-card" style={{ padding: 24 }}>
                            <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-white)', marginBottom: 16 }}>Risk Factors Breakdown</h3>
                            <div className="risk-factors">
                                <div className="risk-factor">
                                    <div className="risk-factor-left">
                                        <div className="risk-factor-icon" style={{ background: 'rgba(239,68,68,0.15)', color: '#f87171' }}>
                                            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>public</span>
                                        </div>
                                        <div>
                                            <div className="risk-factor-name">Cross-border Transaction</div>
                                            <div className="risk-factor-desc">{isCrossBorder ? 'High risk corridor detected' : 'Domestic transaction'}</div>
                                        </div>
                                    </div>
                                    <span className="risk-factor-value" style={{ color: isCrossBorder ? '#ef4444' : '#10b981' }}>
                                        {isCrossBorder ? 'Yes' : 'No'}
                                    </span>
                                </div>

                                <div className="risk-factor">
                                    <div className="risk-factor-left">
                                        <div className="risk-factor-icon" style={{ background: 'rgba(245,158,11,0.15)', color: '#fbbf24' }}>
                                            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>currency_exchange</span>
                                        </div>
                                        <div>
                                            <div className="risk-factor-name">Foreign Currency</div>
                                            <div className="risk-factor-desc">{isForeignCurrency ? 'Currency mismatch with origin' : 'Same currency'}</div>
                                        </div>
                                    </div>
                                    <span className="risk-factor-value" style={{ color: isForeignCurrency ? '#f59e0b' : '#10b981' }}>
                                        {isForeignCurrency ? 'Yes' : 'No'}
                                    </span>
                                </div>

                                <div className="risk-factor">
                                    <div className="risk-factor-left">
                                        <div className="risk-factor-icon" style={{ background: 'rgba(239,68,68,0.15)', color: '#f87171' }}>
                                            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>attach_money</span>
                                        </div>
                                        <div>
                                            <div className="risk-factor-name">Transaction Amount</div>
                                            <div className="risk-factor-desc">Amount analysis</div>
                                        </div>
                                    </div>
                                    <span className="risk-factor-value" style={{ color: 'var(--text-white)' }}>
                                        ${Number(form.amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                    </span>
                                </div>

                                <div className="risk-factor">
                                    <div className="risk-factor-left">
                                        <div className="risk-factor-icon" style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8' }}>
                                            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>bolt</span>
                                        </div>
                                        <div>
                                            <div className="risk-factor-name">Payment Type</div>
                                            <div className="risk-factor-desc">Settlement method analysis</div>
                                        </div>
                                    </div>
                                    <span className="risk-factor-value" style={{ color: 'var(--text-secondary)' }}>
                                        {form.payment_type}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { fetchAccount } from '../api'

function formatMoney(n) {
    if (n == null) return '—'
    if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
    return `$${Number(n).toFixed(2)}`
}

function getRiskClass(cat) {
    return (cat || '').toLowerCase()
}

function getRiskColor(cat) {
    const c = (cat || '').toLowerCase()
    if (c === 'critical') return '#dc2626'
    if (c === 'high') return '#ef4444'
    if (c === 'moderate') return '#f59e0b'
    return '#10b981'
}

function getNodeColor(cat) {
    const c = (cat || '').toLowerCase()
    if (c === 'critical') return '#dc2626'
    if (c === 'high') return '#ef4444'
    if (c === 'moderate') return '#f59e0b'
    if (c === 'low') return '#10b981'
    return '#64748b'
}

function getEdgeColor(cat) {
    const c = (cat || '').toLowerCase()
    if (c === 'critical') return '#dc2626'
    if (c === 'high') return '#ef4444'
    if (c === 'moderate') return '#f59e0b'
    if (c === 'low') return '#10b981'
    return 'rgba(255,255,255,0.2)'
}

function ForceGraph({ graphData, centerAccountId }) {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const tooltipRef = useRef(null)

    useEffect(() => {
        if (!graphData || !graphData.nodes?.length) return

        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const rect = canvas.parentElement.getBoundingClientRect()
        const dpr = window.devicePixelRatio || 1
        canvas.width = rect.width * dpr
        canvas.height = rect.height * dpr
        ctx.scale(dpr, dpr)
        const W = rect.width
        const H = rect.height

        // Deduplicate edges: group by source→target, keep highest risk
        const edgeMap = {}
        for (const e of (graphData.edges || [])) {
            const key = `${e.source}→${e.target}`
            if (!edgeMap[key]) {
                edgeMap[key] = { ...e, txCount: 1, totalAmount: e.amount || 0 }
            } else {
                edgeMap[key].txCount++
                edgeMap[key].totalAmount += (e.amount || 0)
                // keep highest risk
                const riskOrder = { critical: 4, high: 3, moderate: 2, low: 1 }
                const cur = riskOrder[(edgeMap[key].risk_category || '').toLowerCase()] || 0
                const nw = riskOrder[(e.risk_category || '').toLowerCase()] || 0
                if (nw > cur) edgeMap[key].risk_category = e.risk_category
            }
        }
        const dedupedEdges = Object.values(edgeMap)
        const maxAmount = Math.max(...dedupedEdges.map(e => e.totalAmount || 1), 1)

        // Initialize nodes
        const nodeMap = {}
        const nodes = graphData.nodes.map(n => {
            const isCenter = n.id === centerAccountId
            const angle = Math.random() * Math.PI * 2
            const dist = isCenter ? 0 : 80 + Math.random() * 140
            const node = {
                ...n,
                x: W / 2 + Math.cos(angle) * dist,
                y: H / 2 + Math.sin(angle) * dist,
                vx: 0, vy: 0,
                radius: isCenter ? 22 : 10,
                isCenter,
            }
            nodeMap[n.id] = node
            return node
        })

        // Build edges with node references
        const edges = dedupedEdges.map(e => ({
            source: nodeMap[e.source],
            target: nodeMap[e.target],
            risk_category: e.risk_category,
            totalAmount: e.totalAmount,
            txCount: e.txCount,
            payment_type: e.payment_type || '',
            // Normalized width: 1.5 .. 5
            width: 1.5 + (Math.min(e.totalAmount, maxAmount) / maxAmount) * 3.5,
        })).filter(e => e.source && e.target)

        // Check if edges exist between pairs (for curve offset on bidirectional)
        const pairSet = new Set()
        const biDirectional = new Set()
        for (const e of edges) {
            const fwd = `${e.source.id}-${e.target.id}`
            const rev = `${e.target.id}-${e.source.id}`
            if (pairSet.has(rev)) {
                biDirectional.add(fwd)
                biDirectional.add(rev)
            }
            pairSet.add(fwd)
        }

        // Physics simulation
        function simulate() {
            const N = nodes.length
            // Repulsion (all pairs)
            for (let i = 0; i < N; i++) {
                for (let j = i + 1; j < N; j++) {
                    let dx = nodes[j].x - nodes[i].x
                    let dy = nodes[j].y - nodes[i].y
                    let dist = Math.sqrt(dx * dx + dy * dy) || 1
                    let force = 1200 / (dist * dist)
                    let fx = (dx / dist) * force
                    let fy = (dy / dist) * force
                    nodes[i].vx -= fx; nodes[i].vy -= fy
                    nodes[j].vx += fx; nodes[j].vy += fy
                }
            }
            // Attraction along edges
            for (const e of edges) {
                let dx = e.target.x - e.source.x
                let dy = e.target.y - e.source.y
                let dist = Math.sqrt(dx * dx + dy * dy) || 1
                let idealDist = 120
                let force = (dist - idealDist) * 0.008
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                e.source.vx += fx; e.source.vy += fy
                e.target.vx -= fx; e.target.vy -= fy
            }
            // Center gravity + damping
            for (const n of nodes) {
                n.vx += (W / 2 - n.x) * 0.002
                n.vy += (H / 2 - n.y) * 0.002
                n.vx *= 0.88
                n.vy *= 0.88
                if (!n.isCenter) {
                    n.x += n.vx
                    n.y += n.vy
                }
                n.x = Math.max(30, Math.min(W - 30, n.x))
                n.y = Math.max(30, Math.min(H - 30, n.y))
            }
        }

        function drawArrow(ctx, fromX, fromY, toX, toY, targetRadius, color, lineWidth, curved) {
            const dx = toX - fromX
            const dy = toY - fromY
            const len = Math.sqrt(dx * dx + dy * dy) || 1
            const nx = dx / len
            const ny = dy / len

            // Shorten by target radius + arrow head length
            const headLen = 10
            const endX = toX - nx * (targetRadius + 4)
            const endY = toY - ny * (targetRadius + 4)

            ctx.save()
            ctx.strokeStyle = color
            ctx.lineWidth = lineWidth
            ctx.globalAlpha = 0.7

            if (curved) {
                // Curved edge for bidirectional
                const midX = (fromX + endX) / 2
                const midY = (fromY + endY) / 2
                const perpX = -(endY - fromY) / len * 30
                const perpY = (endX - fromX) / len * 30
                const cpX = midX + perpX
                const cpY = midY + perpY

                ctx.beginPath()
                ctx.moveTo(fromX, fromY)
                ctx.quadraticCurveTo(cpX, cpY, endX, endY)
                ctx.stroke()

                // Arrow head on curve end
                // Direction at curve end
                const t = 0.95
                const tangentX = 2 * (1 - t) * (cpX - fromX) + 2 * t * (endX - cpX)
                const tangentY = 2 * (1 - t) * (cpY - fromY) + 2 * t * (endY - cpY)
                const tLen = Math.sqrt(tangentX * tangentX + tangentY * tangentY) || 1
                const tnx = tangentX / tLen
                const tny = tangentY / tLen

                ctx.globalAlpha = 0.9
                ctx.fillStyle = color
                ctx.beginPath()
                ctx.moveTo(endX, endY)
                ctx.lineTo(endX - tnx * headLen + tny * headLen * 0.5, endY - tny * headLen - tnx * headLen * 0.5)
                ctx.lineTo(endX - tnx * headLen - tny * headLen * 0.5, endY - tny * headLen + tnx * headLen * 0.5)
                ctx.closePath()
                ctx.fill()
            } else {
                // Straight edge
                ctx.beginPath()
                ctx.moveTo(fromX, fromY)
                ctx.lineTo(endX, endY)
                ctx.stroke()

                // Arrow head
                ctx.globalAlpha = 0.9
                ctx.fillStyle = color
                ctx.beginPath()
                ctx.moveTo(endX, endY)
                ctx.lineTo(endX - nx * headLen + ny * headLen * 0.5, endY - ny * headLen - nx * headLen * 0.5)
                ctx.lineTo(endX - nx * headLen - ny * headLen * 0.5, endY - ny * headLen + nx * headLen * 0.5)
                ctx.closePath()
                ctx.fill()
            }

            ctx.restore()
        }

        function draw() {
            ctx.clearRect(0, 0, W, H)

            // Draw edges (directed arrows)
            for (const e of edges) {
                const color = getEdgeColor(e.risk_category)
                const pairKey = `${e.source.id}-${e.target.id}`
                const isBidi = biDirectional.has(pairKey)
                drawArrow(ctx, e.source.x, e.source.y, e.target.x, e.target.y,
                    e.target.radius, color, e.width, isBidi)
            }

            // Draw nodes
            for (const n of nodes) {
                const color = getNodeColor(n.risk_category)

                // Outer glow for center node
                if (n.isCenter) {
                    const grd = ctx.createRadialGradient(n.x, n.y, n.radius, n.x, n.y, n.radius + 16)
                    grd.addColorStop(0, `${color}44`)
                    grd.addColorStop(1, `${color}00`)
                    ctx.beginPath()
                    ctx.arc(n.x, n.y, n.radius + 16, 0, Math.PI * 2)
                    ctx.fillStyle = grd
                    ctx.fill()
                }

                // Node circle
                ctx.beginPath()
                ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2)
                ctx.fillStyle = color
                ctx.fill()
                ctx.strokeStyle = n.isCenter ? '#fff' : 'rgba(0,0,0,0.4)'
                ctx.lineWidth = n.isCenter ? 2.5 : 1.5
                ctx.stroke()

                // Label (always show)
                const label = n.isCenter ? '★ THIS' : `...${(n.id || '').slice(-4)}`
                ctx.fillStyle = n.isCenter ? '#fff' : 'rgba(255,255,255,0.7)'
                ctx.font = n.isCenter ? 'bold 11px Inter' : '10px Inter'
                ctx.textAlign = 'center'
                ctx.textBaseline = 'top'
                ctx.fillText(label, n.x, n.y + n.radius + 4)
            }
        }

        // Tooltip on hover
        function handleMouseMove(event) {
            const bounds = canvas.getBoundingClientRect()
            const mx = event.clientX - bounds.left
            const my = event.clientY - bounds.top
            const tooltip = tooltipRef.current
            if (!tooltip) return

            // Check edges
            for (const e of edges) {
                const sx = e.source.x, sy = e.source.y
                const tx = e.target.x, ty = e.target.y
                // Distance from point to line segment
                const dx = tx - sx, dy = ty - sy
                const len2 = dx * dx + dy * dy
                let t = ((mx - sx) * dx + (my - sy) * dy) / (len2 || 1)
                t = Math.max(0, Math.min(1, t))
                const px = sx + t * dx, py = sy + t * dy
                const dist = Math.sqrt((mx - px) ** 2 + (my - py) ** 2)
                if (dist < 8) {
                    const fromLabel = e.source.id === centerAccountId ? 'This Account' : e.source.id
                    const toLabel = e.target.id === centerAccountId ? 'This Account' : e.target.id
                    tooltip.style.display = 'block'
                    tooltip.style.left = `${event.clientX - bounds.left + 12}px`
                    tooltip.style.top = `${event.clientY - bounds.top - 10}px`
                    tooltip.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;color:#fff">${fromLabel} → ${toLabel}</div>
                        <div>Amount: <b>$${Number(e.totalAmount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b></div>
                        <div>Transactions: <b>${e.txCount}</b></div>
                        <div>Risk: <span style="color:${getEdgeColor(e.risk_category)};font-weight:600">${e.risk_category || 'Unknown'}</span></div>
                        ${e.payment_type ? `<div>Type: ${e.payment_type}</div>` : ''}
                    `
                    return
                }
            }

            // Check nodes
            for (const n of nodes) {
                const dist = Math.sqrt((mx - n.x) ** 2 + (my - n.y) ** 2)
                if (dist < n.radius + 4) {
                    tooltip.style.display = 'block'
                    tooltip.style.left = `${event.clientX - bounds.left + 12}px`
                    tooltip.style.top = `${event.clientY - bounds.top - 10}px`
                    tooltip.innerHTML = `
                        <div style="font-weight:600;margin-bottom:4px;color:#fff">${n.id}</div>
                        <div>Risk: <span style="color:${getNodeColor(n.risk_category)};font-weight:600">${n.risk_category || 'Unknown'}</span>
                             (${Number(n.risk_score || 0).toFixed(0)}/100)</div>
                        ${n.total_sent ? `<div>Sent: $${Number(n.total_sent).toLocaleString()}</div>` : ''}
                        ${n.total_received ? `<div>Recv: $${Number(n.total_received).toLocaleString()}</div>` : ''}
                        ${n.isCenter ? '<div style="color:#60a5fa;font-weight:500;margin-top:4px">⭐ Current Account</div>' : ''}
                    `
                    return
                }
            }
            tooltip.style.display = 'none'
        }

        function handleMouseLeave() {
            if (tooltipRef.current) tooltipRef.current.style.display = 'none'
        }

        canvas.addEventListener('mousemove', handleMouseMove)
        canvas.addEventListener('mouseleave', handleMouseLeave)

        let frame = 0
        function tick() {
            simulate()
            draw()
            frame++
            if (frame < 250) animRef.current = requestAnimationFrame(tick)
        }
        tick()

        return () => {
            if (animRef.current) cancelAnimationFrame(animRef.current)
            canvas.removeEventListener('mousemove', handleMouseMove)
            canvas.removeEventListener('mouseleave', handleMouseLeave)
        }
    }, [graphData, centerAccountId])

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
            <div ref={tooltipRef} style={{
                display: 'none',
                position: 'absolute',
                pointerEvents: 'none',
                background: 'rgba(15,23,42,0.95)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: 8,
                padding: '10px 14px',
                fontSize: 12,
                color: '#94a3b8',
                lineHeight: 1.5,
                zIndex: 10,
                backdropFilter: 'blur(8px)',
                maxWidth: 280,
                boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            }} />
        </div>
    )
}



export default function AccountDetail() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        setLoading(true)
        fetchAccount(id)
            .then(d => setData(d))
            .catch(e => setError(e.message))
            .finally(() => setLoading(false))
    }, [id])

    if (loading) return <div className="loading-container"><div className="spinner" /><p>Loading account...</p></div>
    if (error) return <div className="error-container"><span className="material-symbols-outlined" style={{ fontSize: 48 }}>error</span><p>{error}</p></div>
    if (!data) return null

    const acc = data.account || {}
    const txSummary = data.transaction_summary || {}
    const transactions = data.transactions || []
    const graph = data.graph || {}
    const riskColor = getRiskColor(acc.risk_category)
    const score = acc.risk_score != null ? Number(acc.risk_score).toFixed(1) : 0

    return (
        <div className="fade-in">
            <div className="page-header" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <button onClick={() => navigate('/accounts')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                    <span className="material-symbols-outlined">arrow_back</span>
                </button>
                <div>
                    <h2>Account Investigation</h2>
                    <p>Account {id}</p>
                </div>
            </div>

            <div className="detail-layout">
                {/* Left Column */}
                <div>
                    {/* Profile Card */}
                    <div className="glass-card profile-card" style={{ marginBottom: 24 }}>
                        <div className="profile-header">
                            <div className="profile-id">
                                <div className="profile-icon">
                                    <span className="material-symbols-outlined">account_balance</span>
                                </div>
                                <div>
                                    <div className="profile-name">Account</div>
                                    <div className="profile-account-id">ID: {acc.account_id}</div>
                                </div>
                            </div>
                            {/* Risk Gauge */}
                            <div className="risk-gauge" style={{
                                background: `conic-gradient(${riskColor} ${score * 3.6}deg, rgba(255,255,255,0.05) 0deg)`,
                            }}>
                                <div style={{
                                    width: 60, height: 60, borderRadius: '50%', background: 'var(--bg-primary)',
                                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
                                }}>
                                    <span className="risk-gauge-value">{Number(score).toFixed(0)}</span>
                                    <span className="risk-gauge-label" style={{ color: riskColor }}>RISK</span>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginBottom: 16 }}>
                            <span className={`risk-badge ${getRiskClass(acc.risk_category)}`} style={{ fontSize: 14, padding: '4px 14px' }}>
                                ⚠ {acc.risk_category}
                            </span>
                        </div>

                        <div className="stats-grid">
                            <div className="stat-box">
                                <div className="stat-label">Total Sent</div>
                                <div className="stat-value">{formatMoney(acc.total_sent)}</div>
                            </div>
                            <div className="stat-box">
                                <div className="stat-label">Total Received</div>
                                <div className="stat-value">{formatMoney(acc.total_received)}</div>
                            </div>
                            <div className="stat-box">
                                <div className="stat-label">TX Count</div>
                                <div className="stat-value">{(txSummary.total || 0).toLocaleString()}</div>
                            </div>
                            <div className="stat-box">
                                <div className="stat-label">Partners</div>
                                <div className="stat-value">{acc.unique_partners || '—'}</div>
                            </div>
                        </div>

                        <div className="ratio-row">
                            <div className="ratio-box">
                                <span className="ratio-label">Cross-border Ratio</span>
                                <span className="ratio-value">{acc.cross_border_ratio != null ? Number(acc.cross_border_ratio).toFixed(2) : '—'}</span>
                            </div>
                            <div className="ratio-box">
                                <span className="ratio-label">Foreign Currency Ratio</span>
                                <span className="ratio-value">{acc.foreign_currency_ratio != null ? Number(acc.foreign_currency_ratio).toFixed(2) : '—'}</span>
                            </div>
                        </div>
                    </div>

                    {/* Transaction History */}
                    <div className="glass-card" style={{ overflow: 'hidden' }}>
                        <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                            <h3 className="section-title" style={{ margin: 0 }}>Transaction History</h3>
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Counterparty</th>
                                        <th>Amount</th>
                                        <th>Direction</th>
                                        <th>Risk</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {transactions.slice(0, 20).map((tx, i) => {
                                        const isSent = tx.sender_account === id
                                        const counterparty = isSent ? tx.receiver_account : tx.sender_account
                                        return (
                                            <tr key={i}>
                                                <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                                                    {tx.datetime ? new Date(tx.datetime).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '—'}
                                                    <br />
                                                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                                        {tx.datetime ? new Date(tx.datetime).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : ''}
                                                    </span>
                                                </td>
                                                <td style={{ fontWeight: 500, color: 'var(--text-white)', cursor: 'pointer' }}
                                                    onClick={() => navigate(`/accounts/${counterparty}`)}>
                                                    {counterparty}
                                                </td>
                                                <td className="mono">{formatMoney(tx.amount)}</td>
                                                <td>
                                                    <span className={`direction-badge ${isSent ? 'sent' : 'received'}`}>
                                                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                                                            {isSent ? 'north_east' : 'south_west'}
                                                        </span>
                                                        {isSent ? 'Sent' : 'Received'}
                                                    </span>
                                                </td>
                                                <td>
                                                    <span className={`risk-badge ${getRiskClass(tx.prediction_risk_category)}`}>
                                                        {tx.prediction_risk_category || '—'}
                                                    </span>
                                                </td>
                                            </tr>
                                        )
                                    })}
                                    {transactions.length === 0 && (
                                        <tr><td colSpan={5} style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>No transactions found</td></tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Right Column - Network Graph */}
                <div>
                    <div className="glass-card graph-panel">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                            <span className="material-symbols-outlined" style={{ color: 'var(--primary)' }}>hub</span>
                            <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-white)', margin: 0 }}>Transaction Network</h3>
                        </div>
                        <div className="graph-container">
                            <ForceGraph graphData={graph} centerAccountId={id} />
                        </div>
                        <div className="graph-legend">
                            <div className="legend-item"><div className="legend-dot" style={{ background: '#ef4444' }} />High/Critical</div>
                            <div className="legend-item"><div className="legend-dot" style={{ background: '#f59e0b' }} />Moderate</div>
                            <div className="legend-item"><div className="legend-dot" style={{ background: '#10b981' }} />Low Risk</div>
                            <div className="legend-item" style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>arrow_forward</span>
                                Arrow = money flow direction
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

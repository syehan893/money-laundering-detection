'use client'

import { formatNumber, formatPercent, riskColor, type PredictionResponse, type RiskCategory } from '@/lib/api'

const RISK_ORDER: RiskCategory[] = ['Low', 'Moderate', 'High', 'Critical']

/** Confusion matrix 2x2 beserta metrik turunannya. */
function ConfusionMatrix({ result }: { result: PredictionResponse }) {
  const s = result.penilaian
  if (!s.labeled || !s.confusion_matrix) return null
  const { tn, fp, fn, tp } = s.confusion_matrix

  const cells = [
    { label: 'True Negative', value: tn, hint: 'normal, benar dilewatkan', color: 'var(--risk-low)' },
    { label: 'False Positive', value: fp, hint: 'normal, salah ditandai', color: 'var(--risk-moderate)' },
    { label: 'False Negative', value: fn, hint: 'laundering, terlewat', color: 'var(--risk-critical)' },
    { label: 'True Positive', value: tp, hint: 'laundering, tertangkap', color: 'var(--accent)' },
  ]

  return (
    <div className="card">
      <h3>Confusion Matrix</h3>
      <div className="cm-grid">
        {cells.map((c) => (
          <div key={c.label} className="cm-cell" style={{ borderColor: c.color }}>
            <div className="cm-label">{c.label}</div>
            <div className="cm-value" style={{ color: c.color }}>
              {formatNumber(c.value)}
            </div>
            <div className="cm-label" style={{ marginTop: 3 }}>
              {c.hint}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-2" style={{ marginTop: 16, gap: 12 }}>
        {[
          ['Precision', s.precision],
          ['Recall', s.recall],
          ['F1-Score', s.f1],
          ['Accuracy', s.accuracy],
          ['FPR', s.fpr],
          ['FNR', s.fnr],
        ].map(([label, value]) => (
          <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="muted">{label as string}</span>
            <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
              {typeof value === 'number'
                ? label === 'FPR' || label === 'FNR'
                  ? formatPercent(value, 3)
                  : value.toFixed(4)
                : '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Recall per tipologi laundering, diurutkan dari yang paling banyak. */
function TypologyBreakdown({ result }: { result: PredictionResponse }) {
  const per = result.penilaian.per_tipologi
  if (!per || Object.keys(per).length === 0) return null

  const rows = Object.entries(per).sort((a, b) => b[1].total - a[1].total)

  return (
    <div className="card">
      <h3>Recall per Tipologi</h3>
      <table>
        <thead>
          <tr>
            <th>Tipologi</th>
            <th className="num">Jumlah</th>
            <th className="num">Terdeteksi</th>
            <th className="num">Recall</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, v]) => {
            const recall = v.total > 0 ? v.terdeteksi / v.total : 0
            return (
              <tr key={name}>
                <td>{name}</td>
                <td className="num">{formatNumber(v.total)}</td>
                <td className="num">{formatNumber(v.terdeteksi)}</td>
                <td
                  className="num"
                  style={{
                    color:
                      recall >= 0.9 ? 'var(--risk-low)' : recall >= 0.5 ? 'var(--risk-moderate)' : 'var(--risk-critical)',
                    fontWeight: 600,
                  }}
                >
                  {formatPercent(recall, 1)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Daftar transaksi berisiko tertinggi. */
function TopTransactions({ result }: { result: PredictionResponse }) {
  const rows = [...result.prediksi].sort((a, b) => b.probability - a.probability).slice(0, 25)
  const hasLabel = result.penilaian.labeled

  return (
    <div className="card">
      <h3>25 Transaksi Skor Tertinggi</h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Pengirim</th>
              <th>Penerima</th>
              <th className="num">Nominal</th>
              <th className="num">Skor</th>
              <th>Risiko</th>
              {hasLabel && <th>Label Asli</th>}
              {hasLabel && <th>Hasil</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const correct = hasLabel ? (p.flagged ? p.actual_label === 1 : p.actual_label === 0) : null
              return (
                <tr key={p.index}>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.sender}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.receiver}</td>
                  <td className="num">{p.amount.toLocaleString('id-ID', { maximumFractionDigits: 2 })}</td>
                  <td className="num">{p.probability.toFixed(4)}</td>
                  <td>
                    <span
                      className="badge"
                      style={{ background: `${riskColor[p.risk_category]}22`, color: riskColor[p.risk_category] }}
                    >
                      {p.risk_category}
                    </span>
                  </td>
                  {hasLabel && (
                    <td>{p.actual_label === 1 ? (p.laundering_type ?? 'Laundering') : 'Normal'}</td>
                  )}
                  {hasLabel && (
                    <td style={{ color: correct ? 'var(--risk-low)' : 'var(--risk-critical)' }}>
                      {correct ? 'benar' : 'salah'}
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function ResultPanel({ result }: { result: PredictionResponse }) {
  return (
    <div className="grid" style={{ gap: 20 }}>
      <div className="grid grid-4">
        <div className="card">
          <div className="stat-label">Total Transaksi</div>
          <div className="stat-value">{formatNumber(result.total)}</div>
          <div className="stat-hint">dari {result.sumber}</div>
        </div>
        <div className="card">
          <div className="stat-label">Ditandai</div>
          <div className="stat-value" style={{ color: 'var(--risk-critical)' }}>
            {formatNumber(result.ditandai)}
          </div>
          <div className="stat-hint">
            {formatPercent(result.total ? result.ditandai / result.total : 0, 2)} dari total
          </div>
        </div>
        <div className="card">
          <div className="stat-label">Threshold</div>
          <div className="stat-value">{result.threshold?.toFixed(4)}</div>
          <div className="stat-hint">dipilih dari data validasi</div>
        </div>
        <div className="card">
          <div className="stat-label">Waktu Proses</div>
          <div className="stat-value">{result.waktu_proses_detik.toFixed(2)}s</div>
          <div className="stat-hint">
            {result.total > 0
              ? `${Math.round(result.total / Math.max(result.waktu_proses_detik, 0.001)).toLocaleString('id-ID')} tx/detik`
              : '—'}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Distribusi Risiko</h3>
        <div style={{ display: 'flex', height: 34, borderRadius: 8, overflow: 'hidden', gap: 2 }}>
          {RISK_ORDER.map((cat) => {
            const count = result.distribusi_risiko[cat] ?? 0
            const pct = result.total ? (count / result.total) * 100 : 0
            if (pct === 0) return null
            return (
              <div
                key={cat}
                title={`${cat}: ${formatNumber(count)}`}
                style={{
                  width: `${pct}%`,
                  background: riskColor[cat],
                  display: 'grid',
                  placeItems: 'center',
                  fontSize: 12,
                  fontWeight: 600,
                  color: '#ffffff',
                }}
              >
                {pct > 7 ? `${pct.toFixed(0)}%` : ''}
              </div>
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 18, marginTop: 12, flexWrap: 'wrap' }}>
          {RISK_ORDER.map((cat) => (
            <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 13 }}>
              <span
                style={{ width: 10, height: 10, borderRadius: 3, background: riskColor[cat], flex: 'none' }}
              />
              {cat}
              <span className="muted">{formatNumber(result.distribusi_risiko[cat] ?? 0)}</span>
            </div>
          ))}
        </div>
      </div>

      {result.penilaian.labeled ? (
        <div className="grid grid-2">
          <ConfusionMatrix result={result} />
          <TypologyBreakdown result={result} />
        </div>
      ) : (
        <div className="alert alert-info">
          Berkas tidak memuat kolom <code>Is_laundering</code>, sehingga confusion matrix tidak dapat
          dihitung. Hasil prediksi tetap ditampilkan di bawah.
        </div>
      )}

      <TopTransactions result={result} />
    </div>
  )
}

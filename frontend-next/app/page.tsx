'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { api, formatNumber, formatPercent, type ModelInfo } from '@/lib/api'

export default function DashboardPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e: Error) => setError(e.message))
  }, [])

  if (error) {
    return (
      <>
        <div className="page-head">
          <h2>Dashboard</h2>
        </div>
        <div className="alert alert-error">
          Tidak dapat menghubungi backend: {error}
          <div style={{ marginTop: 8 }}>
            Jalankan <code>python -m backend.api</code> lebih dahulu.
          </div>
        </div>
      </>
    )
  }

  if (!info) {
    return (
      <>
        <div className="page-head">
          <h2>Dashboard</h2>
        </div>
        <div className="card">
          <span className="spinner" /> <span style={{ marginLeft: 8 }}>Memuat informasi model…</span>
        </div>
      </>
    )
  }

  const cm = info.confusion_matrix
  const rates = info.error_rates
  const bd = info.breakdown
  const pola = info.per_pola

  return (
    <>
      <div className="page-head">
        <h2>Dashboard</h2>
        <p>Ringkasan model deteksi pencucian uang yang sedang aktif.</p>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="stat-label">Akun Terpantau</div>
          <div className="stat-value">{formatNumber(info.graph.nodes)}</div>
          <div className="stat-hint">simpul dalam graf transaksi</div>
        </div>
        <div className="card">
          <div className="stat-label">Transaksi</div>
          <div className="stat-value">{formatNumber(info.graph.edges)}</div>
          <div className="stat-hint">sisi dalam graf transaksi</div>
        </div>
        <div className="card">
          <div className="stat-label">F1-Score</div>
          <div className="stat-value" style={{ color: 'var(--risk-low)' }}>
            {info.test_metrics ? info.test_metrics.f1.toFixed(4) : '—'}
          </div>
          <div className="stat-hint">pada data uji</div>
        </div>
        <div className="card">
          <div className="stat-label">Versi Model</div>
          <div className="stat-value">{info.version}</div>
          <div className="stat-hint">
            {info.graph.node_features} fitur node · {info.graph.edge_features} fitur edge
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <h3>Hasil pada Data Uji</h3>
          {cm && rates ? (
            <table>
              <tbody>
                <tr>
                  <td className="muted">True Negative</td>
                  <td className="num">{formatNumber(cm.tn)}</td>
                </tr>
                <tr>
                  <td className="muted">False Positive</td>
                  <td className="num">{formatNumber(cm.fp)}</td>
                </tr>
                <tr>
                  <td className="muted">False Negative</td>
                  <td className="num">{formatNumber(cm.fn)}</td>
                </tr>
                <tr>
                  <td className="muted">True Positive</td>
                  <td className="num">{formatNumber(cm.tp)}</td>
                </tr>
                <tr>
                  <td className="muted">False Positive Rate</td>
                  <td className="num" style={{ color: rates.fpr < 0.001 ? 'var(--risk-low)' : 'var(--risk-moderate)' }}>
                    {formatPercent(rates.fpr, 4)}
                  </td>
                </tr>
                <tr>
                  <td className="muted">False Negative Rate</td>
                  <td className="num" style={{ color: rates.fnr < 0.001 ? 'var(--risk-low)' : 'var(--risk-moderate)' }}>
                    {formatPercent(rates.fnr, 4)}
                  </td>
                </tr>
                <tr>
                  <td className="muted">Accuracy</td>
                  <td className="num">{formatPercent(rates.accuracy, 4)}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <div className="muted">Metrik belum tersedia.</div>
          )}
        </div>

        <div className="card">
          <h3>Deteksi per Pola Money Laundering</h3>
          {pola && pola.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>Pola</th>
                  <th className="num">Jumlah</th>
                  <th className="num">Terdeteksi</th>
                  <th className="num">Tidak Terdeteksi</th>
                  <th className="num">Recall</th>
                </tr>
              </thead>
              <tbody>
                {pola.map((p) => (
                  <tr key={p.pola}>
                    <td>{p.label}</td>
                    <td className="num">{formatNumber(p.jumlah)}</td>
                    <td className="num" style={{ color: 'var(--risk-low)' }}>{formatNumber(p.terdeteksi)}</td>
                    <td className="num" style={{ color: 'var(--risk-critical)' }}>{formatNumber(p.tidak_terdeteksi)}</td>
                    <td className="num" style={{ fontWeight: 600 }}>{formatPercent(p.recall, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : bd ? (
            <div className="muted">
              Recall keseluruhan {formatPercent(bd.rare_recall, 2)} ({formatNumber(bd.rare_caught)} dari{' '}
              {formatNumber(bd.rare_total)})
            </div>
          ) : (
            <div className="muted">Rincian belum tersedia.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Klasifikasi Risiko</h3>
        <table>
          <thead>
            <tr>
              <th>Kategori</th>
              <th>Rentang Probabilitas</th>
              <th>Tindakan</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ color: 'var(--risk-low)' }}>Low</td>
              <td>{info.risk_bands.Low}</td>
              <td className="muted">Tidak perlu tindakan</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--risk-moderate)' }}>Moderate</td>
              <td>{info.risk_bands.Moderate}</td>
              <td className="muted">Pantau — tinjau berkala</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--risk-high)' }}>High</td>
              <td>{info.risk_bands.High}</td>
              <td className="muted">Investigasi — eskalasi ke compliance</td>
            </tr>
            <tr>
              <td style={{ color: 'var(--risk-critical)' }}>Critical</td>
              <td>{info.risk_bands.Critical}</td>
              <td className="muted">Blokir — rekomendasi pelaporan SAR</td>
            </tr>
          </tbody>
        </table>
        <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
          <Link href="/production-test" className="btn" style={{ display: 'inline-block' }}>
            Jalankan Uji Produksi
          </Link>
          <Link href="/predict" className="btn btn-ghost" style={{ display: 'inline-block' }}>
            Prediksi Transaksi
          </Link>
        </div>
      </div>
    </>
  )
}

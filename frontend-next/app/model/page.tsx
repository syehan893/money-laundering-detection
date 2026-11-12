'use client'

import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, formatNumber, formatPercent } from '@/lib/api'

interface HistoryPoint {
  epoch: number
  train_loss: number
  val_fnr: number
  val_rare_recall: number
  val_precision: number
  val_recall: number
}

interface Metrics {
  version: string
  best_epoch: number
  optimal_threshold: number
  test_metrics: { precision: number; recall: number; f1: number }
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
  error_rates: { fpr: number; fnr: number; accuracy: number }
  breakdown?: Record<string, number>
  hyperparameters: Record<string, unknown>
  training_history: HistoryPoint[]
}

const CHART_AXIS = '#5b6782'
const CHART_GRID = '#dbe2ee'

export default function ModelPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .metrics()
      .then((d) => setMetrics(d as unknown as Metrics))
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) {
    return (
      <>
        <div className="page-head">
          <h2>Performa Model</h2>
        </div>
        <div className="alert alert-error">{error}</div>
      </>
    )
  }

  if (!metrics) {
    return (
      <>
        <div className="page-head">
          <h2>Performa Model</h2>
        </div>
        <div className="card">
          <span className="spinner" /> <span style={{ marginLeft: 8 }}>Memuat metrik…</span>
        </div>
      </>
    )
  }

  const history = (metrics.training_history ?? []).map((h) => ({
    epoch: h.epoch,
    'Train loss': Number(h.train_loss?.toFixed(5)),
    'Val FNR (%)': Number((h.val_fnr * 100).toFixed(3)),
    'Val Recall (%)': Number(((h.val_recall ?? h.val_rare_recall) * 100).toFixed(2)),
  }))

  const hp = metrics.hyperparameters ?? {}

  return (
    <>
      <div className="page-head">
        <h2>Performa Model</h2>
        <p>
          Metrik pelatihan dan evaluasi model versi {metrics.version}. Threshold{' '}
          {metrics.optimal_threshold?.toFixed(4)} dipilih dari data validasi, lalu diterapkan apa
          adanya pada data uji.
        </p>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        {[
          ['Precision', metrics.test_metrics?.precision],
          ['Recall', metrics.test_metrics?.recall],
          ['F1-Score', metrics.test_metrics?.f1],
          ['Accuracy', metrics.error_rates?.accuracy],
        ].map(([label, value]) => (
          <div className="card" key={label as string}>
            <div className="stat-label">{label as string}</div>
            <div className="stat-value">{typeof value === 'number' ? value.toFixed(4) : '—'}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <h3>Tingkat Kesalahan</h3>
          <table>
            <tbody>
              <tr>
                <td className="muted">False Positive Rate</td>
                <td className="num">{formatPercent(metrics.error_rates.fpr, 4)}</td>
                <td className="num" style={{ color: metrics.error_rates.fpr < 0.001 ? 'var(--risk-low)' : 'var(--risk-moderate)' }}>
                  {metrics.error_rates.fpr < 0.001 ? 'target tercapai' : 'di atas target'}
                </td>
              </tr>
              <tr>
                <td className="muted">False Negative Rate</td>
                <td className="num">{formatPercent(metrics.error_rates.fnr, 4)}</td>
                <td className="num" style={{ color: metrics.error_rates.fnr < 0.001 ? 'var(--risk-low)' : 'var(--risk-moderate)' }}>
                  {metrics.error_rates.fnr < 0.001 ? 'target tercapai' : 'di atas target'}
                </td>
              </tr>
            </tbody>
          </table>

          <h3 style={{ marginTop: 20 }}>Confusion Matrix</h3>
          <div className="cm-grid">
            {[
              ['True Negative', metrics.confusion_matrix.tn, 'var(--risk-low)'],
              ['False Positive', metrics.confusion_matrix.fp, 'var(--risk-moderate)'],
              ['False Negative', metrics.confusion_matrix.fn, 'var(--risk-critical)'],
              ['True Positive', metrics.confusion_matrix.tp, 'var(--accent)'],
            ].map(([label, value, color]) => (
              <div className="cm-cell" key={label as string} style={{ borderColor: color as string }}>
                <div className="cm-label">{label as string}</div>
                <div className="cm-value" style={{ color: color as string }}>
                  {formatNumber(value as number)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Hyperparameter</h3>
          <table>
            <tbody>
              {Object.entries(hp).map(([key, value]) => (
                <tr key={key}>
                  <td className="muted">{key}</td>
                  <td className="num">{String(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {history.length > 0 && (
        <div className="card">
          <h3>Riwayat Pelatihan</h3>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke={CHART_GRID} strokeWidth={0.8} />
                <XAxis
                  dataKey="epoch"
                  stroke={CHART_AXIS}
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fill: CHART_AXIS, fontSize: 12 }}
                />
                <YAxis stroke={CHART_AXIS} tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: '#ffffff',
                    border: '1px solid #dbe2ee',
                    borderRadius: 9,
                    fontSize: 13,
                  }}
                  labelStyle={{ color: '#16213a' }}
                />
                <Line type="monotone" dataKey="Val FNR (%)" stroke="#c8521f" strokeWidth={2} dot={false} />
                <Line
                  type="monotone"
                  dataKey="Val Recall (%)"
                  stroke="#1f64c4"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', gap: 20, marginTop: 10, fontSize: 13 }}>
            <span style={{ color: '#c8521f' }}>■ Val FNR (%)</span>
            <span style={{ color: '#1f64c4' }}>■ Val Recall (%)</span>
            <span className="muted">Epoch terbaik: {metrics.best_epoch}</span>
          </div>
        </div>
      )}
    </>
  )
}

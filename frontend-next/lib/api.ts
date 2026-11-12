/**
 * Klien API untuk backend FastAPI (`backend/api.py`).
 *
 * Seluruh permintaan memakai path relatif `/api/...`; pemetaan ke alamat
 * backend dilakukan oleh rewrite di `next.config.mjs`, sehingga tidak ada
 * masalah CORS saat pengembangan maupun saat di-deploy di belakang proksi.
 */

export type RiskCategory = 'Low' | 'Moderate' | 'High' | 'Critical'

export interface Prediction {
  index: number
  sender: string
  receiver: string
  amount: number
  probability: number
  risk_category: RiskCategory
  flagged: boolean
  actual_label: number | null
  laundering_type: string | null
}

export interface Scoring {
  labeled: boolean
  total: number
  flagged?: number
  confusion_matrix?: { tn: number; fp: number; fn: number; tp: number }
  precision?: number
  recall?: number
  f1?: number
  accuracy?: number
  fpr?: number
  fnr?: number
  per_tipologi?: Record<string, { total: number; terdeteksi: number }>
}

export interface PredictionResponse {
  sumber: string
  total: number
  ditandai: number
  threshold: number
  waktu_proses_detik: number
  distribusi_risiko: Record<RiskCategory, number>
  penilaian: Scoring
  prediksi: Prediction[]
}

export interface ModelInfo {
  version: string
  threshold: number
  device: string
  graph: { nodes: number; edges: number; node_features: number; edge_features: number }
  test_metrics?: { precision: number; recall: number; f1: number }
  confusion_matrix?: { tn: number; fp: number; fn: number; tp: number }
  error_rates?: { fpr: number; fnr: number; accuracy: number }
  breakdown?: {
    rare_total: number
    rare_caught: number
    rare_recall: number
    synthetic_total: number
    synthetic_caught: number
    synthetic_recall: number
  }
  hyperparameters?: Record<string, unknown>
  per_pola?: {
    pola: string
    label: string
    jumlah: number
    terdeteksi: number
    tidak_terdeteksi: number
    recall: number
  }[]
  risk_bands: Record<string, string>
}

export interface TestFileEntry {
  berkas: string
  deskripsi: string
  harapan: string
  baris: number
  laundering: number
  normal: number
  rasio_laundering_persen: number
  tipologi: Record<string, number>
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_version: string
  threshold: number | null
  device: string | null
  error: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', ...init })
  if (!response.ok) {
    let detail: unknown = await response.text()
    try {
      detail = JSON.parse(detail as string)
    } catch {
      /* biarkan sebagai teks */
    }
    const message =
      typeof detail === 'object' && detail !== null && 'detail' in detail
        ? JSON.stringify((detail as { detail: unknown }).detail)
        : String(detail)
    throw new Error(message || `Permintaan gagal (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  modelInfo: () => request<ModelInfo>('/api/model-info'),
  metrics: () => request<Record<string, unknown>>('/api/metrics'),
  graphStats: () => request<Record<string, number>>('/api/graph-stats'),
  testFiles: () =>
    request<{ tersedia: boolean; berkas: TestFileEntry[]; petunjuk?: string }>('/api/test-files'),
  runTestFile: (name: string) =>
    request<PredictionResponse>(`/api/test-files/${encodeURIComponent(name)}`, { method: 'POST' }),
  predict: (transactions: Record<string, string | number>[]) =>
    request<PredictionResponse>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transactions }),
    }),
  predictCsv: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<PredictionResponse>('/api/predict-csv', { method: 'POST', body: form })
  },
}

export const formatNumber = (value: number) => value.toLocaleString('id-ID')

export const formatPercent = (value: number, digits = 2) =>
  `${(value * 100).toLocaleString('id-ID', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`

export const riskColor: Record<RiskCategory, string> = {
  Low: 'var(--risk-low)',
  Moderate: 'var(--risk-moderate)',
  High: 'var(--risk-high)',
  Critical: 'var(--risk-critical)',
}

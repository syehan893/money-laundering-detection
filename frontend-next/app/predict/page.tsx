'use client'

import { useState } from 'react'
import { api, riskColor, type PredictionResponse } from '@/lib/api'

const CURRENCIES = [
  'UK pounds', 'US Dollar', 'Euro', 'Dirham', 'Swiss franc', 'Yen',
  'Turkish lira', 'Mexican Peso', 'Indian rupee', 'Albanian lek',
]
const LOCATIONS = [
  'UK', 'USA', 'UAE', 'Switzerland', 'Germany', 'France', 'Japan',
  'Turkey', 'Mexico', 'India', 'Spain', 'Italy', 'Netherlands', 'Austria',
]
const PAYMENT_TYPES = [
  'Credit card', 'Debit card', 'Cash Deposit', 'Cash Withdrawal',
  'Cheque', 'ACH', 'Cross-border',
]

const today = new Date().toISOString().slice(0, 10)

export default function PredictPage() {
  const [form, setForm] = useState({
    Date: today,
    Time: '14:30:00',
    Sender_account: '',
    Receiver_account: '',
    Amount: '',
    Payment_currency: 'UK pounds',
    Received_currency: 'UK pounds',
    Sender_bank_location: 'UK',
    Receiver_bank_location: 'UK',
    Payment_type: 'Credit card',
  })
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = (key: keyof typeof form, value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }))

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const payload = { ...form, Amount: Number(form.Amount) }
      setResult(await api.predict([payload]))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const prediction = result?.prediksi[0]

  return (
    <>
      <div className="page-head">
        <h2>Prediksi Transaksi</h2>
        <p>
          Masukkan detail satu transaksi untuk memperoleh skor risiko. Model memakai riwayat
          transaksi akun yang bersangkutan sebagai konteks, sehingga akun yang sudah dikenal
          menghasilkan penilaian yang lebih akurat.
        </p>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="grid grid-2">
        <div className="card">
          <h3>Detail Transaksi</h3>
          <form onSubmit={submit}>
            <div className="form-row">
              <div>
                <label htmlFor="sender">Akun Pengirim</label>
                <input
                  id="sender"
                  required
                  value={form.Sender_account}
                  onChange={(e) => update('Sender_account', e.target.value)}
                  placeholder="8724731955"
                />
              </div>
              <div>
                <label htmlFor="receiver">Akun Penerima</label>
                <input
                  id="receiver"
                  required
                  value={form.Receiver_account}
                  onChange={(e) => update('Receiver_account', e.target.value)}
                  placeholder="2769355426"
                />
              </div>
            </div>

            <div className="form-row">
              <div>
                <label htmlFor="amount">Nominal</label>
                <input
                  id="amount"
                  type="number"
                  step="0.01"
                  min="0"
                  required
                  value={form.Amount}
                  onChange={(e) => update('Amount', e.target.value)}
                  placeholder="9500.00"
                />
              </div>
              <div>
                <label htmlFor="ptype">Tipe Pembayaran</label>
                <select id="ptype" value={form.Payment_type} onChange={(e) => update('Payment_type', e.target.value)}>
                  {PAYMENT_TYPES.map((t) => (
                    <option key={t}>{t}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div>
                <label htmlFor="date">Tanggal</label>
                <input id="date" type="date" value={form.Date} onChange={(e) => update('Date', e.target.value)} />
              </div>
              <div>
                <label htmlFor="time">Waktu</label>
                <input id="time" type="time" step="1" value={form.Time} onChange={(e) => update('Time', e.target.value)} />
              </div>
            </div>

            <div className="form-row">
              <div>
                <label htmlFor="pcurr">Mata Uang Pengirim</label>
                <select id="pcurr" value={form.Payment_currency} onChange={(e) => update('Payment_currency', e.target.value)}>
                  {CURRENCIES.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="rcurr">Mata Uang Penerima</label>
                <select id="rcurr" value={form.Received_currency} onChange={(e) => update('Received_currency', e.target.value)}>
                  {CURRENCIES.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div>
                <label htmlFor="sloc">Lokasi Bank Pengirim</label>
                <select id="sloc" value={form.Sender_bank_location} onChange={(e) => update('Sender_bank_location', e.target.value)}>
                  {LOCATIONS.map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="rloc">Lokasi Bank Penerima</label>
                <select id="rloc" value={form.Receiver_bank_location} onChange={(e) => update('Receiver_bank_location', e.target.value)}>
                  {LOCATIONS.map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
              </div>
            </div>

            <button className="btn" type="submit" disabled={loading} style={{ width: '100%', marginTop: 6 }}>
              {loading ? <span className="spinner" /> : 'Hitung Skor Risiko'}
            </button>
          </form>
        </div>

        <div className="card">
          <h3>Hasil Penilaian</h3>
          {!prediction ? (
            <div className="muted" style={{ padding: '40px 0', textAlign: 'center' }}>
              Belum ada hasil. Isi formulir lalu tekan tombol di samping.
            </div>
          ) : (
            <div>
              <div style={{ textAlign: 'center', padding: '18px 0 24px' }}>
                <div
                  style={{
                    fontSize: 46,
                    fontWeight: 700,
                    color: riskColor[prediction.risk_category],
                    lineHeight: 1.1,
                  }}
                >
                  {(prediction.probability * 100).toFixed(1)}%
                </div>
                <div
                  className="badge"
                  style={{
                    background: `${riskColor[prediction.risk_category]}22`,
                    color: riskColor[prediction.risk_category],
                    fontSize: 13,
                    padding: '4px 14px',
                    marginTop: 8,
                  }}
                >
                  Risiko {prediction.risk_category}
                </div>
              </div>

              <table>
                <tbody>
                  <tr>
                    <td className="muted">Probabilitas</td>
                    <td className="num">{prediction.probability.toFixed(6)}</td>
                  </tr>
                  <tr>
                    <td className="muted">Threshold model</td>
                    <td className="num">{result?.threshold?.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td className="muted">Status</td>
                    <td className="num" style={{ color: prediction.flagged ? 'var(--risk-critical)' : 'var(--risk-low)' }}>
                      {prediction.flagged ? 'DITANDAI' : 'lolos'}
                    </td>
                  </tr>
                  <tr>
                    <td className="muted">Lintas negara</td>
                    <td className="num">
                      {form.Sender_bank_location !== form.Receiver_bank_location ? 'ya' : 'tidak'}
                    </td>
                  </tr>
                  <tr>
                    <td className="muted">Beda mata uang</td>
                    <td className="num">
                      {form.Payment_currency !== form.Received_currency ? 'ya' : 'tidak'}
                    </td>
                  </tr>
                  <tr>
                    <td className="muted">Waktu proses</td>
                    <td className="num">{result?.waktu_proses_detik.toFixed(3)} detik</td>
                  </tr>
                </tbody>
              </table>

              <div className="muted" style={{ fontSize: 12.5, marginTop: 14 }}>
                Akun yang belum pernah muncul dalam data historis dinilai tanpa konteks graf,
                sehingga skornya cenderung mendekati nilai tengah.
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

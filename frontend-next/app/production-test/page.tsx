'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import ResultPanel from '@/components/ResultPanel'
import { api, formatNumber, type PredictionResponse, type TestFileEntry } from '@/lib/api'

export default function ProductionTestPage() {
  const [files, setFiles] = useState<TestFileEntry[]>([])
  const [hint, setHint] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [running, setRunning] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api
      .testFiles()
      .then((data) => {
        setFiles(data.berkas ?? [])
        if (!data.tersedia) setHint(data.petunjuk ?? null)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const runBundled = async (name: string) => {
    setRunning(name)
    setError(null)
    try {
      setResult(await api.runTestFile(name))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(null)
    }
  }

  const upload = useCallback(async (file: File) => {
    setRunning(file.name)
    setError(null)
    try {
      setResult(await api.predictCsv(file))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(null)
    }
  }, [])

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file) void upload(file)
  }

  return (
    <>
      <div className="page-head">
        <h2>Uji Produksi</h2>
        <p>
          Unggah berkas CSV transaksi lalu jalankan model terlatih atasnya. Bila berkas memuat kolom{' '}
          <code>Is_laundering</code>, sistem sekaligus menghitung confusion matrix dan recall per
          tipologi.
        </p>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <h3>Unggah Berkas CSV</h3>
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div style={{ fontSize: 15, marginBottom: 6 }}>
              Letakkan berkas di sini atau klik untuk memilih
            </div>
            <div className="muted" style={{ fontSize: 13 }}>
              Format .csv, maksimum 50.000 baris
            </div>
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void upload(file)
                e.target.value = ''
              }}
            />
          </div>
          <div className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>
            Kolom wajib: Time, Date, Sender_account, Receiver_account, Amount, Payment_currency,
            Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type
          </div>
        </div>

        <div className="card">
          <h3>Berkas Uji Bawaan</h3>
          {hint && <div className="alert alert-info">{hint}</div>}
          {files.length === 0 && !hint && <div className="muted">Memuat…</div>}
          <div className="grid" style={{ gap: 10 }}>
            {files.map((f) => (
              <div
                key={f.berkas}
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: 9,
                  padding: '11px 13px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: 12,
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 13.5, fontWeight: 550 }}>{f.berkas}</div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    {formatNumber(f.baris)} baris · {formatNumber(f.laundering)} laundering (
                    {f.rasio_laundering_persen}%)
                  </div>
                </div>
                <button
                  className="btn btn-ghost"
                  style={{ flex: 'none', padding: '7px 13px', fontSize: 13 }}
                  disabled={running !== null}
                  onClick={() => void runBundled(f.berkas)}
                >
                  {running === f.berkas ? <span className="spinner" /> : 'Jalankan'}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {files.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h3>Apa yang Diuji Tiap Berkas</h3>
          <table>
            <thead>
              <tr>
                <th>Berkas</th>
                <th>Isi</th>
                <th>Hasil yang Diharapkan</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.berkas}>
                  <td style={{ whiteSpace: 'nowrap' }}>{f.berkas}</td>
                  <td>{f.deskripsi}</td>
                  <td className="muted">{f.harapan}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {running && !result && (
        <div className="card">
          <span className="spinner" /> <span style={{ marginLeft: 8 }}>Memproses {running}…</span>
        </div>
      )}

      {result && <ResultPanel result={result} />}
    </>
  )
}

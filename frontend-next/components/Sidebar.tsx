'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { api, type HealthResponse } from '@/lib/api'

const LINKS = [
  { href: '/', label: 'Dashboard' },
  { href: '/production-test', label: 'Uji Produksi' },
  { href: '/predict', label: 'Prediksi Transaksi' },
  { href: '/model', label: 'Performa Model' },
]

export default function Sidebar() {
  const pathname = usePathname()
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">A</div>
        <div>
          <h1>AML Guard</h1>
          <span>Deteksi Pencucian Uang</span>
        </div>
      </div>

      <nav>
        {LINKS.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`nav-link ${pathname === link.href ? 'active' : ''}`}
          >
            {link.label}
          </Link>
        ))}
      </nav>

      <div style={{ marginTop: 24, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
        <div className="stat-label">Status Model</div>
        {health === null ? (
          <div className="muted" style={{ fontSize: 13 }}>
            Menghubungi backend…
          </div>
        ) : (
          <div style={{ fontSize: 13 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: health.model_loaded ? 'var(--risk-low)' : 'var(--risk-critical)',
                  flex: 'none',
                }}
              />
              {health.model_loaded ? 'Siap' : 'Model belum dimuat'}
            </div>
            {health.model_loaded && (
              <div className="muted" style={{ fontSize: 12 }}>
                Versi {health.model_version} · {health.device}
                <br />
                Threshold {health.threshold?.toFixed(4)}
              </div>
            )}
            {health.error && (
              <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>
                {health.error.slice(0, 120)}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}

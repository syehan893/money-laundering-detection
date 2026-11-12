import type { Metadata } from 'next'
import './globals.css'
import Sidebar from '@/components/Sidebar'

export const metadata: Metadata = {
  title: 'AML Guard — Deteksi Pencucian Uang',
  description:
    'Sistem deteksi pencucian uang pada transaksi keuangan digital berbasis Graph Attention Network',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  )
}

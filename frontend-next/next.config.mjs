/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proksi ke backend FastAPI supaya frontend cukup memanggil /api/... tanpa CORS
  async rewrites() {
    const backend = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }]
  },
}

export default nextConfig

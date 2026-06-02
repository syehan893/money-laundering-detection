# MongoDB Local Setup for AML Guard

Panduan ini menjelaskan cara menjalankan MongoDB secara lokal agar backend AML Guard dapat terhubung dengan benar.

---

## Prasyarat

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) terinstall dan berjalan
- File `.env` sudah dibuat dari `.env.example`:
  ```bash
  cp .env.example .env
  ```

## Opsi 1: Jalankan Semua Services (MongoDB + Backend + Frontend)

Jika Anda ingin menjalankan seluruh stack (MongoDB, Backend FastAPI, Frontend React):

```bash
docker compose up -d --build
```

Cek status:
```bash
docker compose ps
docker compose logs -f mongodb
docker compose logs -f backend
```

Akses:
- MongoDB: `localhost:27017`
- Backend API: `http://localhost:8000`
- Frontend: `http://localhost`

Hentikan:
```bash
docker compose down
```

---

## Opsi 2: Jalankan MongoDB Saja (Standalone)

Jika Anda ingin menjalankan MongoDB secara terpisah (misal untuk development backend tanpa frontend):

```bash
docker compose -f docker-compose.mongodb.yml up -d
```

Cek status:
```bash
docker compose -f docker-compose.mongodb.yml ps
docker compose -f docker-compose.mongodb.yml logs -f
```

MongoDB akan tersedia di:
- Host: `localhost`
- Port: `27017`
- Username: `admin` (atau sesuai `MONGO_USER` di `.env`)
- Password: sesuai `MONGO_PASS` di `.env`

Hentikan:
```bash
docker compose -f docker-compose.mongodb.yml down
```

---

## Konfigurasi Backend

Backend akan membaca connection string dari environment variable `MONGO_URL`.

### Tanpa Docker (Development Lokal)

Jalankan backend langsung dari Python:

```bash
# Pastikan .env sudah diisi
# Default MONGO_URL sudah menunjuk ke localhost:27017

python -m backend.main
```

### Dengan Docker

Docker Compose secara otomasuk mengatur `MONGO_URL` ke container MongoDB:

```yaml
MONGO_URL=mongodb://${MONGO_USER}:${MONGO_PASS}@mongodb:27017/${MONGO_DB}?authSource=admin
```

---

## Verifikasi Koneksi

Setelah MongoDB berjalan, test koneksi dengan healthcheck endpoint:

```bash
curl http://localhost:8000/api/health
```

Response yang diharapkan:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "model_loaded": true
}
```

Atau test langsung dengan Python:

```python
from backend.database import get_sync_db
db = get_sync_db()
print(db.list_collection_names())
```

---

## Troubleshooting

### `Connection refused` ke MongoDB

1. Pastikan container MongoDB berjalan:
   ```bash
   docker ps | grep mongo
   ```

2. Jika menggunakan Docker Compose full stack, pastikan backend menunggu MongoDB healthy:
   ```bash
   docker compose logs backend | grep "MongoDB"
   ```

3. Jika jalankan standalone, pastikan `MONGO_URL` di `.env` mengarah ke `localhost`:
   ```
   MONGO_URL=mongodb://admin:admin123@localhost:27017/money-laundering-detection?authSource=admin
   ```

### Port 27017 sudah digunakan

Ubah port di `.env`:
```
MONGO_PORT=27018
```
Kemudian update `MONGO_URL`:
```
MONGO_URL=mongodb://admin:admin123@localhost:27018/money-laundering-detection?authSource=admin
```

### Data MongoDB persisten

Data MongoDB disimpan di Docker Volume `mongodb_data`. Volume ini tidak akan hilang meski container dihapus.

Untuk reset data:
```bash
docker compose down -v  # menghapus volume
docker compose up -d    # membuat volume baru
```

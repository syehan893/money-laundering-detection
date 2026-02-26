"""
database.py — MongoDB Connection & Collection Access
======================================================
Provides both synchronous (pymongo) and asynchronous (motor) clients:

    - Sync client  → used by train.py to populate data after training
    - Async client → used by main.py (FastAPI) to serve API requests

Collections:
    - accounts:          Account profiles with risk scores
    - transactions:      All transactions with predictions
    - training_metrics:  Training results snapshot (1 doc per run)
    - predictions:       Real-time predictions from /api/predict
"""

import os

from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

# ─── Connection ───────────────────────────────────────────────────────────────
MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb://admin:Krowten200S*@syehan.xyz:27017/money-laundering-detection?authSource=admin",
)
DB_NAME = os.getenv("MONGO_DB", "money-laundering-detection")


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNCHRONOUS CLIENT (for train.py)
# ═══════════════════════════════════════════════════════════════════════════════

_sync_client: MongoClient = None
_sync_db = None


def get_sync_db():
    """Get synchronous MongoDB database reference (pymongo)."""
    global _sync_client, _sync_db
    if _sync_db is None:
        print("  🔄 Connecting to MongoDB (sync) ...")
        _sync_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
        _sync_db = _sync_client[DB_NAME]
        # Test connection
        info = _sync_client.server_info()
        print(f"  ✓ MongoDB connected (v{info.get('version', '?')})")
        print(f"  ✓ Database: {DB_NAME}")
    return _sync_db


def close_sync_db():
    """Close synchronous MongoDB connection."""
    global _sync_client, _sync_db
    if _sync_client:
        _sync_client.close()
        _sync_client = None
        _sync_db = None
        print("  MongoDB (sync) disconnected.")


def setup_sync_indexes(db):
    """Create indexes for fast queries (sync)."""
    # Accounts
    db.accounts.create_index("account_id", unique=True)
    db.accounts.create_index("risk_category")
    db.accounts.create_index("risk_score")

    # Transactions
    db.transactions.create_index("sender_account")
    db.transactions.create_index("receiver_account")
    db.transactions.create_index("prediction_risk_category")
    db.transactions.create_index("datetime")

    # Training metrics
    db.training_metrics.create_index("run_id")

    # Predictions log
    db.predictions.create_index("created_at")

    print("  ✓ Indexes created")


# ═══════════════════════════════════════════════════════════════════════════════
#  ASYNCHRONOUS CLIENT (for main.py / FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

_async_client: AsyncIOMotorClient = None
_async_db = None


async def connect_async_db():
    """Connect to MongoDB asynchronously (motor) for FastAPI."""
    global _async_client, _async_db
    print("  🔄 Connecting to MongoDB (async) ...")
    _async_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=10000)
    _async_db = _async_client[DB_NAME]

    # Test connection
    info = await _async_client.server_info()
    print(f"  ✓ MongoDB connected (v{info.get('version', '?')})")
    print(f"  ✓ Database: {DB_NAME}")

    # Ensure indexes
    await _async_db.accounts.create_index("account_id", unique=True)
    await _async_db.accounts.create_index("risk_category")
    await _async_db.accounts.create_index("risk_score")
    await _async_db.transactions.create_index("sender_account")
    await _async_db.transactions.create_index("receiver_account")
    await _async_db.transactions.create_index("prediction_risk_category")
    await _async_db.transactions.create_index("datetime")
    await _async_db.training_metrics.create_index("run_id")
    await _async_db.predictions.create_index("created_at")

    return _async_db


async def close_async_db():
    """Close async MongoDB connection."""
    global _async_client, _async_db
    if _async_client:
        _async_client.close()
        _async_client = None
        _async_db = None
        print("  MongoDB (async) disconnected.")


def get_async_db():
    """Get async database reference."""
    return _async_db

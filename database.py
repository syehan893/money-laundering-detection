"""
database.py — MongoDB Connection & Collection Access
======================================================
Async MongoDB driver (motor) for the AML Detection API.

Collections:
    - accounts:      Account profiles with risk scores
    - transactions:  All transactions with predictions
    - model_metrics: Training results snapshot
"""

import os

from motor.motor_asyncio import AsyncIOMotorClient

# ─── Connection ───────────────────────────────────────────────────────────────
# tlsInsecure=true in the URL bypasses SSL cert issues on MS Store Python
MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb+srv://syehanart:bSmsqyodB8crjqkv@cluster0.yo9ng.mongodb.net/?appName=Cluster0&tls=true&tlsInsecure=true",
)
DB_NAME = os.getenv("MONGO_DB", "aml_detection")

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    """Connect to MongoDB and create indexes."""
    global client, db
    print("  🔄 Connecting to MongoDB ...")
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]

    # Create indexes for fast queries
    await db.accounts.create_index("account_id", unique=True)
    await db.accounts.create_index("risk_category")
    await db.accounts.create_index("risk_score")
    await db.transactions.create_index("sender_account")
    await db.transactions.create_index("receiver_account")
    await db.transactions.create_index("prediction_risk_category")
    await db.transactions.create_index("datetime")

    # Test connection
    info = await client.server_info()
    print(f"  ✓ MongoDB connected (v{info.get('version', '?')})")
    print(f"  ✓ Database: {DB_NAME}")
    return db


async def close_db():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("  MongoDB disconnected.")


def get_db():
    """Get database reference."""
    return db

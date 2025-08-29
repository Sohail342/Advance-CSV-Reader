import asyncio
from app.utils.database import engine, Base
from app.models.csv_headers import CSVHeaders

async def init_database():
    """Initialize the database with all tables."""
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_database())
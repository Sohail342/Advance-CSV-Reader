from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Use an async database URL (example uses PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./csv_data.db")
print(DATABASE_URL)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


# Async DB dependency for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

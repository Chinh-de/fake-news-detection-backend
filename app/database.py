import os
import ssl 
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.config import settings

DATABASE_URL = settings.DATABASE_URL

# 1. Tự động nhận diện môi trường: Kiểm tra xem URL có trỏ về máy local không
is_local = "localhost" in DATABASE_URL or "127.0.0.1" in DATABASE_URL

connect_args = {}

# 2. Chỉ cấu hình SSL nếu KHÔNG PHẢI môi trường local (chạy trên Aiven/Supabase)
if not is_local:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_context
else:
    print("--- 🔌 Đang kết nối tới Database LOCAL: Tự động tắt SSL ---")

# Create database engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
    connect_args=connect_args  # <--- Cấu hình này giờ sẽ tự động rỗng {} khi chạy Local
)

# Async session factory
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Declarative base
Base = declarative_base()

# Dependency to get db session in endpoints
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
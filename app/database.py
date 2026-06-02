import ssl # <--- 1. Import thêm thư viện ssl chuẩn của Python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.config import settings

# 2. Khởi tạo cấu hình SSL context cho thư viện asyncpg
# Cách này giúp bỏ qua việc cấu hình chuỗi mã hóa phức tạp trên DATABASE_URL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Create database engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
    connect_args={
        "ssl": ssl_context  # <--- 3. Truyền bộ cấu hình SSL vào đây
    }
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
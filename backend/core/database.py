"""
数据库配置
包含数据库连接、会话管理和依赖注入
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from backend.models.base import Base

# 数据库配置
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///autoclip.db"
)

# 如果没有设置环境变量，使用配置函数获取数据库URL
if DATABASE_URL == "sqlite:///autoclip.db":
    try:
        from .config import get_database_url
        DATABASE_URL = get_database_url()
    except ImportError:
        # 如果导入失败，保持默认值
        pass

# 创建数据库引擎
if "sqlite" in DATABASE_URL:
    sqlite_options = {
        "connect_args": {
            "check_same_thread": False,
            "timeout": 30,
        },
        "pool_pre_ping": True,
        "echo": False,
    }
    # StaticPool is only safe for an in-memory SQLite database. A desktop file
    # database is used by API and worker threads concurrently, so each checked
    # out session must have its own connection and transaction.
    if ":memory:" in DATABASE_URL:
        sqlite_options["poolclass"] = StaticPool
    engine = create_engine(DATABASE_URL, **sqlite_options)

    @event.listens_for(engine, "connect")
    def _configure_sqlite_connection(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL配置
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False
    )

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db() -> Generator[Session, None, None]:
    """
    数据库会话依赖注入
    用于FastAPI的依赖注入系统
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """创建所有数据库表"""
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """删除所有数据库表"""
    Base.metadata.drop_all(bind=engine)

def reset_database():
    """重置数据库"""
    drop_tables()
    create_tables()

from sqlalchemy import text

def test_connection() -> bool:
    """测试数据库连接"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).fetchone()
        return True
    except Exception as e:
        print(f"数据库连接测试失败: {e}")
        return False

# 数据库初始化
def init_database():
    """初始化数据库"""
    print("正在初始化数据库...")
    
    # 测试连接
    if not test_connection():
        print("❌ 数据库连接失败")
        return False
    
    # 创建表
    try:
        create_tables()
        print("✅ 数据库表创建成功")
        return True
    except Exception as e:
        print(f"❌ 数据库表创建失败: {e}")
        return False

if __name__ == "__main__":
    # 直接运行此文件时初始化数据库
    init_database()

"""SQLAlchemy engine、session、declarative base。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from sla.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


# SQLite 专用参数:check_same_thread=False 让 FastAPI 多线程访问安全
engine_kwargs = {}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 依赖注入:每个请求一个 session。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

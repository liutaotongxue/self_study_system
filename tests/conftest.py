"""全项目 DB 测试隔离(O4 起,首个写库测试族;后续 DB 测试受益)。

每测试一个临时文件 sqlite + Base.metadata.create_all,零触真 app.db。
无需 rollback 体操;库函数已不自持 commit(O4 Δ1),flush 后 close 即弃。
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sla.db import Base
import sla.models.domain  # noqa: F401  注册 Domain/Document/Chunk/Note/Question 到 Base.metadata
import sla.models.kg      # noqa: F401  注册 KGNode/KGEdge


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()

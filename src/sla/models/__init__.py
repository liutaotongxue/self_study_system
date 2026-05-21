"""ORM 模型。导入子模块让 alembic 能发现所有 model。"""
from sla.models import domain, kg, runtime  # noqa: F401

"""ORM models. Import submodules so Alembic can discover every model."""
from sla.models import domain, kg, runtime  # noqa: F401

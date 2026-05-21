"""Alembic environment configuration."""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 导入所有 model 让 alembic 能发现 metadata
from sla.db import Base
from sla.config import settings
from sla.models import domain, runtime  # noqa: F401  导入触发 model 注册

config = context.config
# 单一真源:alembic 与 app 读同一 settings.database_url,杜绝
# "alembic.ini 写死 ./app.db vs .env 改了 DATABASE_URL" 的静默分叉
# (setup 假成功 + app 起来无表)。offline/online 两路经 set_main_option 全覆盖。
config.set_main_option("sqlalchemy.url", settings.database_url)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite 友好,允许 ALTER TABLE
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

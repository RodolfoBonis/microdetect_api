import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Importar a classe Base e o objeto engine da aplicação
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from microdetect.database.database import Base
from microdetect.core import settings

# Importar todos os modelos para que o Alembic os detecte

# isto é o objeto Alembic Config, que fornece
# acesso aos valores dentro do arquivo .ini em uso.
config = context.config

# Substituir a configuração de URL do SQLAlchemy com a da aplicação
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpretar o arquivo de configuração do alembic.ini para configuração Python
# de logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Adicionar MetaData do seu modelo aqui
# para suporte 'autogenerate'
target_metadata = Base.metadata

# outras opções de contexto passadas:

def run_migrations_offline() -> None:
    """Executa migrações em modo 'offline'.

    Este configura o contexto com apenas uma URL
    e não um Engine, embora um Engine seja aceitável
    aqui também. Ao chamar engine.connect(), o contexto em si
    conectará e executará a função dada.

    """
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
    """Executa migrações em modo 'online'.

    Neste cenário, criamos um Engine
    e associa uma conexão a ele.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() 
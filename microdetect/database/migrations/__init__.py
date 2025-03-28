# Em microdetect/database/migrations/__init__.py
import logging
from pathlib import Path

from microdetect.core.config import settings

logger = logging.getLogger(__name__)


def get_migrations_dir():
    """Retorna o diretório de migrações do Alembic."""
    return Path(__file__).parent


def apply_migrations():
    """Aplica migrações pendentes ao banco de dados."""
    migrations_dir = get_migrations_dir()

    try:
        # Importar alembic se disponível
        import alembic
        from alembic.config import Config
        from alembic import command

        # Criar configuração Alembic
        alembic_cfg = Config(str(migrations_dir / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

        # Executar migração
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrações aplicadas com sucesso")
        return True
    except ImportError:
        logger.warning("Alembic não encontrado, migrações não serão aplicadas")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar migrações: {e}")
        return False


def create_migration(message):
    """Cria uma nova migração."""
    migrations_dir = get_migrations_dir()

    try:
        # Importar alembic
        import alembic
        from alembic.config import Config
        from alembic import command

        # Criar configuração Alembic
        alembic_cfg = Config(str(migrations_dir / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

        # Criar migração
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"Migração '{message}' criada com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar migração: {e}")
        return False
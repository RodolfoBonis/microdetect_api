# Em microdetect/database/migrations/__init__.py
import logging
import sys
import traceback
from pathlib import Path

from microdetect.core.config import settings

logger = logging.getLogger(__name__)


def get_migrations_dir():
    """Retorna o diretório de migrações do Alembic."""
    return Path(__file__).parent


def get_alembic_config_path():
    """Retorna o caminho para o arquivo alembic.ini."""
    # Primeiro, tenta encontrar no diretório raiz do projeto
    project_root = Path(__file__).parent.parent.parent.parent
    alembic_ini_path = project_root / "alembic.ini"
    
    if alembic_ini_path.exists():
        logger.info(f"Arquivo alembic.ini encontrado em: {alembic_ini_path}")
        return alembic_ini_path
    
    # Se não encontrar no diretório raiz, tenta dentro do diretório de migrações
    migrations_dir = get_migrations_dir()
    alembic_ini_path = migrations_dir / "alembic.ini"
    
    if alembic_ini_path.exists():
        logger.info(f"Arquivo alembic.ini encontrado em: {alembic_ini_path}")
        return alembic_ini_path
    
    # Se não encontrar em nenhum lugar, loga um aviso e retorna None
    logger.warning(f"Arquivo alembic.ini não encontrado")
    return None


def apply_migrations():
    """Aplica migrações pendentes ao banco de dados."""
    migrations_dir = get_migrations_dir()
    alembic_config_path = get_alembic_config_path()
    
    if not alembic_config_path:
        logger.warning("Arquivo alembic.ini não encontrado, migrações não serão aplicadas")
        return False

    try:
        logger.info("Tentando importar alembic...")
        # Verificar se alembic pode ser importado
        import importlib.util
        spec = importlib.util.find_spec("alembic")
        if spec is None:
            logger.error("Módulo alembic não encontrado no sys.path")
            logger.info(f"Caminhos de busca atuais (sys.path): {sys.path}")
            return False
        
        # Importar alembic se disponível
        logger.info("Importando alembic...")
        import alembic
        logger.info(f"Alembic importado com sucesso: versão {alembic.__version__}")
        
        from alembic.config import Config
        from alembic import command

        # Criar configuração Alembic
        logger.info(f"Criando configuração com arquivo: {alembic_config_path}")
        alembic_cfg = Config(str(alembic_config_path))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

        # Executar migração
        logger.info("Executando migração (upgrade to 'head')...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrações aplicadas com sucesso")
        return True
    except ImportError as e:
        logger.error(f"Erro ao importar alembic: {e}")
        logger.error(f"Detalhes: {traceback.format_exc()}")
        logger.warning("Alembic não encontrado, migrações não serão aplicadas")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar migrações: {e}")
        logger.error(f"Detalhes: {traceback.format_exc()}")
        return False


def create_migration(message):
    """Cria uma nova migração."""
    migrations_dir = get_migrations_dir()
    alembic_config_path = get_alembic_config_path()
    
    if not alembic_config_path:
        logger.warning("Arquivo alembic.ini não encontrado, não é possível criar migrações")
        return False

    try:
        # Importar alembic
        import alembic
        from alembic.config import Config
        from alembic import command

        # Criar configuração Alembic
        alembic_cfg = Config(str(alembic_config_path))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

        # Criar migração
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"Migração '{message}' criada com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar migração: {e}")
        logger.error(f"Detalhes: {traceback.format_exc()}")
        return False
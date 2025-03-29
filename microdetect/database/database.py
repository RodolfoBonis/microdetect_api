from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from microdetect.core.config import settings
import logging

# Configurar o logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Criar engine do SQLAlchemy
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}  # Necessário apenas para SQLite
)

# Criar sessão local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Criar base para os modelos
Base = declarative_base()

def verify_tables_exist():
    """Verificar se todas as tabelas existem e criar se não existirem."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # Importar todos os modelos explicitamente para registrar no Base.metadata
    from microdetect.models.dataset import Dataset
    from microdetect.models.image import Image
    from microdetect.models.annotation import Annotation
    from microdetect.models.model import Model
    from microdetect.models.training_session import TrainingSession
    from microdetect.models.dataset_image import DatasetImage
    from microdetect.models.inference_result import InferenceResult
    
    # Verificar se todas as tabelas definidas nos modelos existem
    missing_tables = set()
    for table_name in Base.metadata.tables.keys():
        if table_name not in tables:
            missing_tables.add(table_name)
    
    if missing_tables:
        logger.warning(f"Tabelas faltando no banco de dados: {missing_tables}")
        logger.info("Criando tabelas faltantes...")
        # Criar apenas as tabelas faltantes
        Base.metadata.create_all(engine)
        logger.info("Tabelas criadas com sucesso.")
    else:
        logger.info("Todas as tabelas existem no banco de dados.")
    
    return True

def get_db():
    """Função para obter uma sessão do banco de dados."""
    # Verificar se as tabelas existem antes de criar a sessão
    verify_tables_exist()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Inicializa o banco de dados e aplica migrações."""
    # Criar conexão com banco de dados
    engine = create_engine(settings.DATABASE_URL)
    
    logger.info("Inicializando banco de dados...")

    try:
        # Executar migrações Alembic
        logger.info("Tentando aplicar migrações Alembic...")
        from microdetect.database.migrations import apply_migrations
        if apply_migrations():
            logger.info("Migrações Alembic aplicadas com sucesso")
        else:
            logger.warning("Falha ao aplicar migrações Alembic, criando tabelas diretamente")
            Base.metadata.create_all(bind=engine)
            logger.info("Tabelas criadas diretamente com SQLAlchemy")
    except Exception as e:
        logger.error(f"Erro ao aplicar migrações: {e}")
        # Fallback: criar tabelas diretamente
        logger.info("Fallback: criando tabelas diretamente com SQLAlchemy")
        Base.metadata.create_all(bind=engine)

    return engine
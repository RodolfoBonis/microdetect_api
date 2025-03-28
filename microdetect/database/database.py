from sqlalchemy import create_engine
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

def get_db():
    """Função para obter uma sessão do banco de dados."""
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
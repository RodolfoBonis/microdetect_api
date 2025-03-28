from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from microdetect.core.config import settings
import logging

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

    try:
        # Executar migrações Alembic
        from microdetect.database.migrations import apply_migrations
        apply_migrations()
    except Exception as e:
        logger.error(f"Erro ao aplicar migrações: {e}")
        # Fallback: criar tabelas diretamente
        Base.metadata.create_all(bind=engine)

    return engine
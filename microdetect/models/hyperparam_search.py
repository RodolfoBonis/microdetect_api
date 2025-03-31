from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from microdetect.database.database import Base

class HyperparamSearchStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class HyperparamSearch(Base):
    __tablename__ = "hyperparam_searches"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500))
    status = Column(Enum(HyperparamSearchStatus), default=HyperparamSearchStatus.PENDING)
    search_space = Column(JSON)  # Espaço de busca de hiperparâmetros
    best_params = Column(JSON)   # Melhores hiperparâmetros encontrados
    best_metrics = Column(JSON)  # Métricas com os melhores hiperparâmetros
    iterations = Column(Integer, default=5)  # Número de iterações de busca
    trials_data = Column(JSON)  # Dados de todas as tentativas
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Chave estrangeira para dataset
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relacionamentos
    dataset = relationship("Dataset")
    
    # Para facilitar o acesso ao modelo de treinamento criado com os melhores parâmetros
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"), nullable=True)
    training_session = relationship("TrainingSession") 
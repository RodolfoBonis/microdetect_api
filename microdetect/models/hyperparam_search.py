from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from microdetect.models.base import BaseModel

class HyperparamSearchStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class HyperparamSearch(BaseModel):
    __tablename__ = "hyperparam_searches"

    name = Column(String(255), nullable=False)
    description = Column(String(500))
    status = Column(String(9))  # Usando os mesmos valores do enum HyperparamSearchStatus
    search_space = Column(JSON)  # Espaço de busca de hiperparâmetros
    best_params = Column(JSON)   # Melhores hiperparâmetros encontrados
    best_metrics = Column(JSON)  # Métricas com os melhores hiperparâmetros
    iterations = Column(Integer)
    trials_data = Column(JSON)  # Dados de todas as tentativas
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Chave estrangeira para dataset
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Para facilitar o acesso ao modelo de treinamento criado com os melhores parâmetros
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"))
    
    # Relacionamentos
    dataset = relationship("Dataset", back_populates="hyperparam_searches")
    training_session = relationship("TrainingSession", back_populates="hyperparam_search") 
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from microdetect.database.database import Base

class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500))
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING)
    model_type = Column(String(50))  # Tipo do modelo (ex: "yolov8")
    model_version = Column(String(50))  # Versão do modelo
    hyperparameters = Column(JSON)  # Parâmetros de treinamento
    metrics = Column(JSON)  # Métricas de treinamento (loss, accuracy, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Chave estrangeira para dataset
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relacionamentos
    dataset = relationship("Dataset", back_populates="training_sessions")
    model = relationship("Model", back_populates="training_session", uselist=False) 
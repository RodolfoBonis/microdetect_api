from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500))
    filepath = Column(String(512), nullable=False)  # Caminho para o arquivo do modelo
    model_type = Column(String(50))  # Tipo do modelo (ex: "yolov8")
    model_version = Column(String(50))  # Versão do modelo
    metrics = Column(JSON)  # Métricas de validação
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Chave estrangeira para sessão de treinamento
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"))
    
    # Relacionamentos
    training_session = relationship("TrainingSession", back_populates="model")
    inference_results = relationship("InferenceResult", back_populates="model") 
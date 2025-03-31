from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class TrainingReport(Base):
    __tablename__ = "training_reports"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    metrics_history = Column(JSON)  # Lista de métricas por época
    confusion_matrix = Column(JSON)  # Matriz de confusão
    class_performance = Column(JSON)  # Desempenho por classe
    final_metrics = Column(JSON)  # Métricas finais do modelo
    resource_usage_avg = Column(JSON)  # Uso médio de recursos
    resource_usage_max = Column(JSON)  # Uso máximo de recursos
    hyperparameters = Column(JSON)  # Hiperparâmetros utilizados
    train_images_count = Column(Integer)  # Número de imagens de treino
    val_images_count = Column(Integer)    # Número de imagens de validação
    test_images_count = Column(Integer)   # Número de imagens de teste
    training_time_seconds = Column(Integer)  # Tempo total de treinamento
    model_size_mb = Column(Float)  # Tamanho do modelo em MB
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Chaves estrangeiras
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relacionamentos
    training_session = relationship("TrainingSession", back_populates="reports")
    dataset = relationship("Dataset") 
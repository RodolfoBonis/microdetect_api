from sqlalchemy import Column, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    predictions = Column(JSON)  # Lista de detecções com bounding boxes e confianças
    metrics = Column(JSON)  # Métricas de inferência (tempo, FPS, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Chaves estrangeiras
    image_id = Column(Integer, ForeignKey("images.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    
    # Relacionamentos
    image = relationship("Image", back_populates="inference_results")
    model = relationship("Model", back_populates="inference_results") 
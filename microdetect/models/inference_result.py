from sqlalchemy import Column, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from microdetect.models.base import BaseModel

class InferenceResult(BaseModel):
    __tablename__ = "inference_results"

    predictions = Column(JSON)  # Lista de detecções com bounding boxes e confianças
    metrics = Column(JSON)  # Métricas de inferência (tempo, FPS, etc.)
    
    # Chaves estrangeiras
    image_id = Column(Integer, ForeignKey("images.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    
    # Relacionamentos
    image = relationship("Image", back_populates="inference_results")
    model = relationship("Model", back_populates="inference_results") 
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from microdetect.models.base import BaseModel

class Image(BaseModel):
    __tablename__ = "images"

    file_name = Column(String(255), index=True)
    file_path = Column(String(255), unique=True)
    file_size = Column(Integer)  # Tamanho em bytes
    url = Column(String(255))  # URL para acessar a imagem via API
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    image_metadata = Column(JSON)  # Metadados da imagem (resolução, formato, ajustes, etc.)
    
    # Relacionamentos
    dataset_images = relationship("DatasetImage", back_populates="image_ref")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    inference_results = relationship("InferenceResult", back_populates="image") 
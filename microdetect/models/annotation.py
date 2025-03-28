from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float)  # Confiança da anotação (0-1)
    bbox = Column(JSON)  # Bounding box [x, y, width, height]
    
    # Campos explícitos para as coordenadas e dimensões do bounding box
    x = Column(Float, nullable=True)  # Coordenada x do canto superior esquerdo
    y = Column(Float, nullable=True)  # Coordenada y do canto superior esquerdo
    width = Column(Float, nullable=True)  # Largura do bounding box
    height = Column(Float, nullable=True)  # Altura do bounding box
    area = Column(Float, nullable=True)  # Área do bounding box (calculada como width * height)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Chaves estrangeiras
    image_id = Column(Integer, ForeignKey("images.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relacionamentos
    image = relationship("Image", back_populates="annotations")
    dataset = relationship("Dataset", back_populates="annotations") 
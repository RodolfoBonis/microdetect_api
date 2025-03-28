from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), index=True)
    file_path = Column(String(255), unique=True)
    file_size = Column(Integer)  # Tamanho em bytes
    url = Column(String(255))  # URL para acessar a imagem via API
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    image_metadata = Column(JSON)  # Metadados da imagem (resolução, formato, ajustes, etc.)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    dataset = relationship("Dataset", back_populates="images")
    dataset_images = relationship("DatasetImage", backref="image_ref")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    inference_results = relationship("InferenceResult", back_populates="image") 
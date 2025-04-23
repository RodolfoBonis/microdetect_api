from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base
from microdetect.models.base import BaseModel

class Dataset(BaseModel):
    """Modelo para datasets."""
    __tablename__ = "datasets"

    name = Column(String(255), nullable=False)
    description = Column(Text)
    path = Column(String, nullable=False)
    classes = Column(JSON, default=list)  # Lista de classes no formato JSON
    
    # Relacionamentos
    dataset_images = relationship("DatasetImage", back_populates="dataset", cascade="all, delete-orphan")
    training_sessions = relationship("TrainingSession", back_populates="dataset", cascade="all, delete-orphan")
    hyperparam_searches = relationship("HyperparamSearch", back_populates="dataset", cascade="all, delete-orphan")
    
    # Acesso direto às imagens associadas
    images = relationship(
        "Image",
        secondary="dataset_images",
        backref="datasets",
        viewonly=True
    )
    
    annotations = relationship("Annotation", back_populates="dataset") 
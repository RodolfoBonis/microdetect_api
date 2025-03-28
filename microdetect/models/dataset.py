from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from microdetect.database.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    classes = Column(JSON, default=list)  # Lista de classes no formato JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    images = relationship("Image", back_populates="dataset")
    
    # Nova relação N:N através da tabela de associação
    dataset_images = relationship("DatasetImage", backref="dataset_ref")
    
    # Acesso direto às imagens associadas
    associated_images = relationship(
        "Image",
        secondary="dataset_images",
        backref="associated_datasets",
        viewonly=True
    )
    
    annotations = relationship("Annotation", back_populates="dataset")
    training_sessions = relationship("TrainingSession", back_populates="dataset") 
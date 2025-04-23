from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from microdetect.models.base import BaseModel

class DatasetImage(BaseModel):
    """Modelo para imagens de datasets."""
    __tablename__ = "dataset_images"

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    path = Column(String, nullable=False)
    
    # Relacionamentos
    dataset = relationship("Dataset", back_populates="dataset_images")
    image_ref = relationship("Image", back_populates="dataset_images")
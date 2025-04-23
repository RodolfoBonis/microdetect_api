from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from microdetect.models.base import BaseModel

class Annotation(BaseModel):
    __tablename__ = "annotations"

    class_name = Column(String(100), nullable=False)
    confidence = Column(Float)  # Confiança da anotação (0-1)
    bbox = Column(JSON)  # Bounding box [x, y, width, height]
    
    # Campos explícitos para as coordenadas e dimensões do bounding box
    x = Column(Float, nullable=True)  # Coordenada x do canto superior esquerdo
    y = Column(Float, nullable=True)  # Coordenada y do canto superior esquerdo
    width = Column(Float, nullable=True)  # Largura do bounding box
    height = Column(Float, nullable=True)  # Altura do bounding box
    area = Column(Float, nullable=True)  # Área do bounding box (calculada como width * height)
    
    # Chaves estrangeiras
    image_id = Column(Integer, ForeignKey("images.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relacionamentos
    image = relationship("Image", back_populates="annotations")
    dataset = relationship("Dataset", back_populates="annotations")
    
    @property
    def bounding_box(self):
        """
        Property que retorna o bounding box no formato esperado pelo schema.
        Usa campos x, y, width, height diretamente se bbox não estiver disponível.
        """
        if self.bbox is not None:
            return self.bbox
        elif self.x is not None and self.y is not None and self.width is not None and self.height is not None:
            return {
                'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height
            }
        else:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0} 
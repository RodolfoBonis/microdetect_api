from sqlalchemy import Column, Integer, DateTime, ForeignKey
from datetime import datetime
from microdetect.database.database import Base

# Tabela de associação entre datasets e imagens
class DatasetImage(Base):
    __tablename__ = "dataset_images"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
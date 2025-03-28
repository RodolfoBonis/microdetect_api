from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any, List

# Importar o schema de dataset (evitando referência circular)
class DatasetSummary(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True

class ImageBase(BaseModel):
    file_name: str
    file_path: str
    file_size: Optional[int] = None
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_metadata: Optional[Dict[str, Any]] = None
    dataset_id: Optional[int] = None

class ImageCreate(ImageBase):
    file_name: str
    file_path: str
    image_metadata: Optional[Dict[str, Any]] = None
    dataset_id: Optional[int] = None

class ImageUpdate(BaseModel):
    file_name: Optional[str] = None
    image_metadata: Optional[Dict[str, Any]] = None
    dataset_id: Optional[int] = None

class ImageResponse(ImageBase):
    id: int
    created_at: datetime
    updated_at: datetime
    # Adicionar lista de datasets associados
    datasets: Optional[List[DatasetSummary]] = []

    class Config:
        from_attributes = True 
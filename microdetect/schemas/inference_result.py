from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any

class InferenceResultBase(BaseModel):
    predictions: List[Dict[str, Any]]  # Lista de detecções
    metrics: Dict[str, Any]  # Métricas de inferência
    image_id: int
    model_id: int

class InferenceResultCreate(InferenceResultBase):
    pass

class InferenceResultResponse(InferenceResultBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True 
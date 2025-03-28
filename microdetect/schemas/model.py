from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    filepath: str
    model_type: str
    model_version: str
    metrics: Optional[Dict[str, Any]] = None
    training_session_id: int

class ModelCreate(ModelBase):
    pass

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class ModelResponse(ModelBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 
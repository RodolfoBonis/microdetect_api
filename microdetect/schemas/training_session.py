from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
from microdetect.models.training_session import TrainingStatus

class TrainingSessionBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: str
    model_version: str
    hyperparameters: Optional[Dict[str, Any]] = None
    dataset_id: int

class TrainingSessionCreate(TrainingSessionBase):
    pass

class TrainingSessionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TrainingStatus] = None
    metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TrainingSessionResponse(TrainingSessionBase):
    id: int
    status: TrainingStatus
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True 
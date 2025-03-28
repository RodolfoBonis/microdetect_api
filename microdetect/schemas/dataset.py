from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    classes: Optional[List[str]] = []

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(DatasetBase):
    name: Optional[str] = None
    classes: Optional[List[str]] = None

class ClassDistribution(BaseModel):
    class_name: str
    count: int
    percentage: float

class DatasetResponse(DatasetBase):
    id: int
    created_at: datetime
    updated_at: datetime
    images_count: Optional[int] = 0
    annotations_count: Optional[int] = 0
    class_distribution: Optional[List[ClassDistribution]] = None

    class Config:
        from_attributes = True 
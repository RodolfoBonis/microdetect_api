from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnnotationBase(BaseModel):
    bounding_box: Dict[str, float]  # x, y, width, height
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnnotationCreate(AnnotationBase):
    image_id: int
    dataset_id: Optional[int] = None

class AnnotationBatchItem(AnnotationBase):
    pass

class AnnotationBatch(BaseModel):
    image_id: int
    dataset_id: Optional[int] = None
    annotations: List[AnnotationBatchItem]

class AnnotationUpdate(AnnotationBase):
    image_id: Optional[int] = None
    dataset_id: Optional[int] = None
    class_name: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnnotationResponse(AnnotationBase):
    id: int
    image_id: int
    dataset_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 
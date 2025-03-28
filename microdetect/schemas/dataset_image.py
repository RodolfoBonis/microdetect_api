from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DatasetImageBase(BaseModel):
    dataset_id: int
    image_id: int

class DatasetImageCreate(DatasetImageBase):
    pass

class DatasetImageResponse(DatasetImageBase):
    id: int
    created_at: datetime
    message: Optional[str] = None

    class Config:
        from_attributes = True 
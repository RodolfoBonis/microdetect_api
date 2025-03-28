from typing import Optional
from pydantic import BaseModel

class ClassInfo(BaseModel):
    class_name: str
    count: int
    percentage: float
    is_used: bool = False
    is_undefined: bool = False
    
    class Config:
        from_attributes = True

class ClassDistributionResponse(BaseModel):
    class_name: str
    count: int
    percentage: float
    is_used: bool = False
    is_undefined: bool = False
    
    class Config:
        from_attributes = True 
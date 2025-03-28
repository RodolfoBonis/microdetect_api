from typing import Optional
from microdetect.schemas.base import BaseSchema

class ClassInfo(BaseSchema):
    def __init__(self,
                class_name: str,
                count: int,
                percentage: float,
                is_used: bool = False,
                is_undefined: bool = False):
        super().__init__(
            class_name=class_name,
            count=count,
            percentage=percentage,
            is_used=is_used,
            is_undefined=is_undefined
        )
    
    @classmethod
    def from_orm(cls, obj):
        return cls(
            class_name=obj.class_name,
            count=obj.count,
            percentage=obj.percentage,
            is_used=getattr(obj, 'is_used', False),
            is_undefined=getattr(obj, 'is_undefined', False)
        )

class ClassDistributionResponse(BaseSchema):
    def __init__(self,
                class_name: str,
                count: int,
                percentage: float,
                is_used: bool = False,
                is_undefined: bool = False):
        super().__init__(
            class_name=class_name,
            count=count,
            percentage=percentage,
            is_used=is_used,
            is_undefined=is_undefined
        )
    
    @classmethod
    def from_orm(cls, obj):
        return cls(
            class_name=obj.class_name,
            count=obj.count,
            percentage=obj.percentage,
            is_used=getattr(obj, 'is_used', False),
            is_undefined=getattr(obj, 'is_undefined', False)
        ) 
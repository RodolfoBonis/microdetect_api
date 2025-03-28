from datetime import datetime
from typing import Optional, List, Dict, Any
from microdetect.schemas.base import BaseSchema

class ClassDistribution(BaseSchema):
    def __init__(self, class_name: str, count: int, percentage: float):
        super().__init__(
            class_name=class_name,
            count=count,
            percentage=percentage
        )

class DatasetBase(BaseSchema):
    def __init__(self, 
                name: str, 
                description: Optional[str] = None, 
                classes: Optional[List[str]] = None):
        super().__init__(
            name=name,
            description=description,
            classes=classes if classes is not None else []
        )

class DatasetCreate(DatasetBase):
    """Classe para criação de um dataset."""
    pass

class DatasetUpdate(BaseSchema):
    def __init__(self,
                name: Optional[str] = None,
                description: Optional[str] = None,
                classes: Optional[List[str]] = None):
        super().__init__(
            name=name,
            description=description,
            classes=classes
        )

class DatasetResponse(DatasetBase):
    def __init__(self,
                id: int,
                name: str,
                created_at: datetime,
                updated_at: datetime,
                description: Optional[str] = None,
                classes: Optional[List[str]] = None,
                images_count: Optional[int] = 0,
                annotations_count: Optional[int] = 0,
                class_distribution: Optional[List[ClassDistribution]] = None):
        super().__init__(
            name=name,
            description=description,
            classes=classes
        )
        self.id = id
        self.created_at = created_at
        self.updated_at = updated_at
        self.images_count = images_count
        self.annotations_count = annotations_count
        self.class_distribution = class_distribution if class_distribution is not None else []
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        # Processar class_distribution se existir
        class_dist = None
        if hasattr(obj, 'class_distribution') and obj.class_distribution:
            if isinstance(obj.class_distribution, list):
                class_dist = [
                    ClassDistribution(
                        class_name=item.class_name,
                        count=item.count,
                        percentage=item.percentage
                    ) for item in obj.class_distribution
                ]
            elif isinstance(obj.class_distribution, dict):
                class_dist = [
                    ClassDistribution(
                        class_name=k,
                        count=v.get('count', 0),
                        percentage=v.get('percentage', 0.0)
                    ) for k, v in obj.class_distribution.items()
                ]
        
        return cls(
            id=obj.id,
            name=obj.name,
            description=getattr(obj, 'description', None),
            classes=getattr(obj, 'classes', []),
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            images_count=getattr(obj, 'images_count', 0),
            annotations_count=getattr(obj, 'annotations_count', 0),
            class_distribution=class_dist
        ) 
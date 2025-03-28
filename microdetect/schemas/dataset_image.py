from datetime import datetime
from typing import Optional
from microdetect.schemas.base import BaseSchema

class DatasetImageBase(BaseSchema):
    def __init__(self, dataset_id: int, image_id: int):
        super().__init__(
            dataset_id=dataset_id,
            image_id=image_id
        )

class DatasetImageCreate(DatasetImageBase):
    """Classe para criação de associação entre dataset e imagem."""
    pass

class DatasetImageResponse(DatasetImageBase):
    def __init__(self,
                id: int,
                dataset_id: int,
                image_id: int,
                created_at: datetime,
                message: Optional[str] = None):
        super().__init__(
            dataset_id=dataset_id,
            image_id=image_id
        )
        self.id = id
        self.created_at = created_at
        self.message = message
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        return cls(
            id=obj.id,
            dataset_id=obj.dataset_id,
            image_id=obj.image_id,
            created_at=obj.created_at,
            message=getattr(obj, 'message', None)
        ) 
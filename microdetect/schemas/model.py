from datetime import datetime
from typing import Optional, Dict, Any
from microdetect.schemas.base import BaseSchema

class ModelBase(BaseSchema):
    def __init__(self,
                name: str,
                filepath: str,
                model_type: str,
                model_version: str,
                training_session_id: int,
                description: Optional[str] = None,
                metrics: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            description=description,
            filepath=filepath,
            model_type=model_type,
            model_version=model_version,
            metrics=metrics if metrics is not None else {},
            training_session_id=training_session_id
        )

class ModelCreate(ModelBase):
    """Classe para criação de um modelo."""
    pass

class ModelUpdate(BaseSchema):
    def __init__(self,
                name: Optional[str] = None,
                description: Optional[str] = None,
                metrics: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            description=description,
            metrics=metrics
        )

class ModelResponse(ModelBase):
    def __init__(self,
                id: int,
                created_at: datetime,
                updated_at: datetime,
                name: str,
                filepath: str,
                model_type: str,
                model_version: str,
                training_session_id: int,
                description: Optional[str] = None,
                metrics: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            filepath=filepath,
            model_type=model_type,
            model_version=model_version,
            training_session_id=training_session_id,
            description=description,
            metrics=metrics
        )
        self.id = id
        self.created_at = created_at
        self.updated_at = updated_at
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        return cls(
            id=obj.id,
            name=obj.name,
            description=getattr(obj, 'description', None),
            filepath=obj.filepath,
            model_type=obj.model_type,
            model_version=obj.model_version,
            metrics=getattr(obj, 'metrics', {}),
            training_session_id=obj.training_session_id,
            created_at=obj.created_at,
            updated_at=obj.updated_at
        ) 
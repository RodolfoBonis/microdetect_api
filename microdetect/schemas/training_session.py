from datetime import datetime
from typing import Optional, Dict, Any
from microdetect.models.training_session import TrainingStatus
from microdetect.schemas.base import BaseSchema

class TrainingSessionBase(BaseSchema):
    def __init__(self,
                name: str,
                model_type: str,
                model_version: str,
                dataset_id: int,
                description: Optional[str] = None,
                hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            description=description,
            model_type=model_type,
            model_version=model_version,
            hyperparameters=hyperparameters if hyperparameters is not None else {},
            dataset_id=dataset_id
        )

class TrainingSessionCreate(TrainingSessionBase):
    """Classe para criação de sessão de treinamento."""
    pass

class TrainingSessionUpdate(BaseSchema):
    def __init__(self,
                name: Optional[str] = None,
                description: Optional[str] = None,
                status: Optional[TrainingStatus] = None,
                metrics: Optional[Dict[str, Any]] = None,
                started_at: Optional[datetime] = None,
                completed_at: Optional[datetime] = None):
        super().__init__(
            name=name,
            description=description,
            status=status,
            metrics=metrics,
            started_at=started_at,
            completed_at=completed_at
        )

class TrainingSessionResponse(TrainingSessionBase):
    def __init__(self,
                id: int,
                status: TrainingStatus,
                created_at: datetime,
                updated_at: datetime,
                name: str,
                model_type: str,
                model_version: str,
                dataset_id: int,
                description: Optional[str] = None,
                hyperparameters: Optional[Dict[str, Any]] = None,
                metrics: Optional[Dict[str, Any]] = None,
                started_at: Optional[datetime] = None,
                completed_at: Optional[datetime] = None):
        super().__init__(
            name=name,
            model_type=model_type,
            model_version=model_version,
            dataset_id=dataset_id,
            description=description,
            hyperparameters=hyperparameters
        )
        self.id = id
        self.status = status
        self.metrics = metrics if metrics is not None else {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.started_at = started_at
        self.completed_at = completed_at
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        return cls(
            id=obj.id,
            name=obj.name,
            description=getattr(obj, 'description', None),
            model_type=obj.model_type,
            model_version=obj.model_version,
            dataset_id=obj.dataset_id,
            hyperparameters=getattr(obj, 'hyperparameters', {}),
            status=obj.status,
            metrics=getattr(obj, 'metrics', {}),
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            started_at=getattr(obj, 'started_at', None),
            completed_at=getattr(obj, 'completed_at', None)
        ) 
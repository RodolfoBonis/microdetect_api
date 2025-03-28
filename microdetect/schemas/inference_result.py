from datetime import datetime
from typing import List, Dict, Any
from microdetect.schemas.base import BaseSchema

class InferenceResultBase(BaseSchema):
    def __init__(self,
                predictions: List[Dict[str, Any]],  # Lista de detecções
                metrics: Dict[str, Any],  # Métricas de inferência
                image_id: int,
                model_id: int):
        super().__init__(
            predictions=predictions,
            metrics=metrics,
            image_id=image_id,
            model_id=model_id
        )

class InferenceResultCreate(InferenceResultBase):
    """Classe para criação de resultados de inferência."""
    pass

class InferenceResultResponse(InferenceResultBase):
    def __init__(self,
                id: int,
                created_at: datetime,
                predictions: List[Dict[str, Any]],
                metrics: Dict[str, Any],
                image_id: int,
                model_id: int):
        super().__init__(
            predictions=predictions,
            metrics=metrics,
            image_id=image_id,
            model_id=model_id
        )
        self.id = id
        self.created_at = created_at
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        return cls(
            id=obj.id,
            predictions=obj.predictions,
            metrics=obj.metrics,
            image_id=obj.image_id,
            model_id=obj.model_id,
            created_at=obj.created_at
        ) 
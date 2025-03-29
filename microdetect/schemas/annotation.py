from typing import Optional, List, Dict, Any
from datetime import datetime
from microdetect.schemas.base import BaseSchema

class AnnotationBase(BaseSchema):
    def __init__(self,
                bounding_box: Dict[str, float],  # x, y, width, height
                class_name: Optional[str] = None,
                confidence: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            bounding_box=bounding_box,
            class_name=class_name,
            confidence=confidence,
            metadata=metadata if metadata is not None else {}
        )

class AnnotationCreate(AnnotationBase):
    def __init__(self,
                bounding_box: Dict[str, float],
                image_id: int,
                dataset_id: Optional[int] = None,
                class_name: Optional[str] = None,
                confidence: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            bounding_box=bounding_box,
            class_name=class_name,
            confidence=confidence,
            metadata=metadata
        )
        self.image_id = image_id
        self.dataset_id = dataset_id

class AnnotationBatchItem(AnnotationBase):
    """Item para criação em lote"""
    pass

class AnnotationBatch(BaseSchema):
    def __init__(self,
                image_id: int,
                annotations: List[AnnotationBatchItem],
                dataset_id: Optional[int] = None):
        super().__init__(
            image_id=image_id,
            dataset_id=dataset_id,
            annotations=annotations
        )

class AnnotationUpdate(BaseSchema):
    def __init__(self,
                image_id: Optional[int] = None,
                dataset_id: Optional[int] = None,
                class_name: Optional[str] = None,
                bounding_box: Optional[Dict[str, float]] = None,
                confidence: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            image_id=image_id,
            dataset_id=dataset_id,
            class_name=class_name,
            bounding_box=bounding_box,
            confidence=confidence,
            metadata=metadata
        )

class AnnotationResponse(AnnotationBase):
    def __init__(self,
                id: int,
                image_id: int,
                bounding_box: Dict[str, float],
                created_at: datetime,
                updated_at: datetime,
                dataset_id: Optional[int] = None,
                class_name: Optional[str] = None,
                confidence: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            bounding_box=bounding_box,
            class_name=class_name,
            confidence=confidence,
            metadata=metadata
        )
        self.id = id
        self.image_id = image_id
        self.dataset_id = dataset_id
        self.created_at = created_at
        self.updated_at = updated_at
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        # Construir bounding_box a partir de x, y, width, height se bbox não estiver disponível
        if hasattr(obj, 'bbox') and obj.bbox is not None:
            bounding_box = obj.bbox
        elif hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'width') and hasattr(obj, 'height'):
            bounding_box = {
                'x': obj.x,
                'y': obj.y,
                'width': obj.width,
                'height': obj.height
            }
        else:
            # Fallback para evitar erro
            bounding_box = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            
        return cls(
            id=obj.id,
            image_id=obj.image_id,
            dataset_id=getattr(obj, 'dataset_id', None),
            bounding_box=bounding_box,
            class_name=getattr(obj, 'class_name', None),
            confidence=getattr(obj, 'confidence', None),
            metadata=getattr(obj, 'metadata', {}),
            created_at=obj.created_at,
            updated_at=obj.updated_at
        ) 
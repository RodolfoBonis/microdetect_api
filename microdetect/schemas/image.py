from datetime import datetime
from typing import Optional, Dict, Any, List
from microdetect.schemas.base import BaseSchema

# Classe de resumo do dataset (evitando referência circular)
class DatasetSummary(BaseSchema):
    def __init__(self, id: int, name: str, description: Optional[str] = None):
        super().__init__(
            id=id,
            name=name,
            description=description
        )
    
    @classmethod
    def from_orm(cls, obj):
        return cls(
            id=obj.id,
            name=obj.name,
            description=obj.description
        )

# Import da classe de anotação aqui para evitar importação circular
from microdetect.schemas.annotation import AnnotationResponse

class ImageBase(BaseSchema):
    def __init__(self, 
                file_name: str,
                file_path: str,
                file_size: Optional[int] = None,
                url: Optional[str] = None,
                width: Optional[int] = None,
                height: Optional[int] = None,
                image_metadata: Optional[Dict[str, Any]] = None,
                dataset_id: Optional[int] = None):
        super().__init__(
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            url=url,
            width=width,
            height=height,
            image_metadata=image_metadata if image_metadata is not None else {},
            dataset_id=dataset_id
        )

class ImageCreate(ImageBase):
    def __init__(self,
                file_name: str,
                file_path: str,
                image_metadata: Optional[Dict[str, Any]] = None,
                dataset_id: Optional[int] = None):
        super().__init__(
            file_name=file_name,
            file_path=file_path,
            image_metadata=image_metadata,
            dataset_id=dataset_id
        )

class ImageUpdate(BaseSchema):
    def __init__(self,
                file_name: Optional[str] = None,
                image_metadata: Optional[Dict[str, Any]] = None,
                dataset_id: Optional[int] = None):
        super().__init__(
            file_name=file_name,
            image_metadata=image_metadata,
            dataset_id=dataset_id
        )

class ImageResponse(ImageBase):
    def __init__(self,
                id: int,
                created_at: datetime,
                updated_at: datetime,
                file_name: str,
                file_path: str,
                file_size: Optional[int] = None,
                url: Optional[str] = None,
                width: Optional[int] = None,
                height: Optional[int] = None,
                image_metadata: Optional[Dict[str, Any]] = None,
                dataset_id: Optional[int] = None,
                datasets: Optional[List[DatasetSummary]] = None,
                annotations: Optional[List[AnnotationResponse]] = None):
        super().__init__(
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            url=url,
            width=width,
            height=height,
            image_metadata=image_metadata,
            dataset_id=dataset_id
        )
        self.id = id
        self.created_at = created_at
        self.updated_at = updated_at
        self.datasets = datasets if datasets is not None else []
        self.annotations = annotations if annotations is not None else []
    
    @classmethod
    def from_orm(cls, obj):
        # Converter objeto ORM para classe de resposta
        datasets = []
        if hasattr(obj, 'datasets') and obj.datasets:
            datasets = [DatasetSummary.from_orm(ds) for ds in obj.datasets]
        
        # Converter anotações se existirem
        annotations = []
        if hasattr(obj, 'annotations') and obj.annotations:
            from microdetect.schemas.annotation import AnnotationResponse
            annotations = [AnnotationResponse.from_orm(ann) for ann in obj.annotations]
            
        return cls(
            id=obj.id,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            file_name=obj.file_name,
            file_path=obj.file_path,
            file_size=getattr(obj, 'file_size', None),
            url=getattr(obj, 'url', None),
            width=getattr(obj, 'width', None),
            height=getattr(obj, 'height', None),
            image_metadata=getattr(obj, 'image_metadata', {}),
            dataset_id=getattr(obj, 'dataset_id', None),
            datasets=datasets,
            annotations=annotations
        ) 
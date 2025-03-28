from typing import List, Optional, Dict, Any
from datetime import datetime
from microdetect.schemas.base import BaseSchema

class SimpleModelBase(BaseSchema):
    def __init__(self, name: str, description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name, 
            description=description,
            parameters=parameters if parameters is not None else {}
        )

class SimpleModelCreate(SimpleModelBase):
    """Classe para criação de um modelo simples."""
    pass

class SimpleModelUpdate(BaseSchema):
    def __init__(self, 
                name: Optional[str] = None, 
                description: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None,
                is_active: Optional[bool] = None):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            is_active=is_active
        )

class SimpleModelResponse(SimpleModelBase):
    def __init__(self,
                id: int,
                name: str,
                description: Optional[str] = None,
                parameters: Optional[Dict[str, Any]] = None,
                created_at: datetime = None,
                updated_at: datetime = None,
                is_active: bool = True):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters
        )
        self.id = id
        self.created_at = created_at
        self.updated_at = updated_at
        self.is_active = is_active
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        return cls(
            id=obj.id,
            name=obj.name,
            description=getattr(obj, 'description', None),
            parameters=getattr(obj, 'parameters', {}),
            created_at=getattr(obj, 'created_at', None),
            updated_at=getattr(obj, 'updated_at', None),
            is_active=getattr(obj, 'is_active', True)
        ) 
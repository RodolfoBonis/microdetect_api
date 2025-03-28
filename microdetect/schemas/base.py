from typing import Dict, Any, ClassVar, Optional, Type

class BaseSchema:
    """Classe base para todos os esquemas, fornecendo funcionalidades comuns."""
    
    def __init__(self, **kwargs):
        """Inicializa o objeto com os atributos fornecidos."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def dict(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Converte o objeto para um dicionário, similar ao método do Pydantic."""
        result = {}
        for key, value in self.__dict__.items():
            if not exclude_unset or value is not None:
                if hasattr(value, 'dict') and callable(getattr(value, 'dict')):
                    result[key] = value.dict()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_orm(cls, obj: Any) -> 'BaseSchema':
        """Converte um objeto ORM para o esquema."""
        # Implementação padrão para classes que não sobrescrevem este método
        data = {}
        for column in getattr(obj, '__table__').columns:
            data[column.name] = getattr(obj, column.name)
        return cls(**data) 
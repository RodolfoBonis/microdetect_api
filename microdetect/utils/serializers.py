import json
from datetime import datetime, date
from typing import Any, Dict, List, Union

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Python types."""

    def default(self, obj):
        # Handle datetime
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # Handle Enum classes
        if hasattr(obj, '__members__'):
            return str(obj)

        # Handle Enum values
        if hasattr(obj, 'value') and hasattr(obj, 'name') and hasattr(type(obj), '__members__'):
            return obj.value

        # Try to get a dictionary
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)

def serialize_to_json(obj: Any) -> str:
    """Converte qualquer objeto Python em uma string JSON."""
    return json.dumps(obj, cls=JSONEncoder)

def serialize_to_dict(obj: Any) -> Dict:
    """Converte qualquer objeto Python em um dicionário."""
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        # Se o objeto tiver um método dict(), usar esse método
        return obj.dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif isinstance(obj, list):
        return [serialize_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return str(obj)

def build_response(data: Any) -> Dict:
    """Cria uma resposta padronizada para a API."""
    return {
        "data": serialize_to_dict(data),
        "success": True,
        "timestamp": datetime.now().isoformat()
    }

def build_error_response(message: str, status_code: int = 400) -> Dict:
    """Cria uma resposta de erro padronizada para a API."""
    return {
        "error": message,
        "success": False,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    } 
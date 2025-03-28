from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List

class DatasetStatistics(BaseModel):
    """
    Modelo que representa estatísticas detalhadas de um dataset
    """
    total_images: int = Field(..., description="Número total de imagens no dataset")
    total_annotations: int = Field(..., description="Número total de anotações (objetos marcados) no dataset")
    annotated_images: int = Field(..., description="Número de imagens que têm pelo menos uma anotação")
    unannotated_images: int = Field(..., description="Número de imagens sem nenhuma anotação")
    
    average_image_size: Optional[Dict[str, Any]] = Field(None, description="Tamanho médio das imagens em pixels (largura x altura)")
    object_size_distribution: Optional[Dict[str, int]] = Field(None, description="Distribuição de tamanhos de objetos anotados (pequeno, médio, grande)")
    class_imbalance: Optional[float] = Field(None, description="Desbalanceamento entre classes (quanto maior, mais desbalanceado)")
    average_objects_per_image: Optional[float] = Field(None, description="Número médio de objetos por imagem")
    average_object_density: Optional[float] = Field(None, description="Densidade média de objetos (objetos por área de imagem)")
    last_calculated: Optional[datetime] = Field(None, description="Último cálculo das estatísticas (timestamp)")
    class_counts: Optional[Dict[str, int]] = Field(None, description="Contagem detalhada por classe")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="Dados extras específicos da aplicação")
    
    class Config:
        schema_extra = {
            "example": {
                "total_images": 100,
                "total_annotations": 450,
                "annotated_images": 80,
                "unannotated_images": 20,
                "average_image_size": {"width": 640, "height": 480},
                "object_size_distribution": {"small": 150, "medium": 250, "large": 50},
                "class_imbalance": 0.25,
                "average_objects_per_image": 4.5,
                "average_object_density": 0.015,
                "last_calculated": "2025-03-27T12:00:00",
                "class_counts": {"cell": 200, "bacteria": 250},
                "extra_data": {"quality_score": 0.85}
            }
        } 
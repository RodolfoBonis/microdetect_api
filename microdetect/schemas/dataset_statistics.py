from datetime import datetime
from typing import Optional, Dict, Any, List
from microdetect.schemas.base import BaseSchema

class DatasetStatistics(BaseSchema):
    """
    Modelo que representa estatísticas detalhadas de um dataset
    """
    def __init__(self,
                total_images: int,  # Número total de imagens no dataset
                total_annotations: int,  # Número total de anotações (objetos marcados) no dataset
                annotated_images: int,  # Número de imagens que têm pelo menos uma anotação
                unannotated_images: int,  # Número de imagens sem nenhuma anotação
                average_image_size: Optional[Dict[str, Any]] = None,  # Tamanho médio das imagens em pixels (largura x altura)
                object_size_distribution: Optional[Dict[str, int]] = None,  # Distribuição de tamanhos de objetos anotados (pequeno, médio, grande)
                class_imbalance: Optional[float] = None,  # Desbalanceamento entre classes (quanto maior, mais desbalanceado)
                average_objects_per_image: Optional[float] = None,  # Número médio de objetos por imagem
                average_object_density: Optional[float] = None,  # Densidade média de objetos (objetos por área de imagem)
                last_calculated: Optional[datetime] = None,  # Último cálculo das estatísticas (timestamp)
                class_counts: Optional[Dict[str, int]] = None,  # Contagem detalhada por classe
                extra_data: Optional[Dict[str, Any]] = None):  # Dados extras específicos da aplicação
        super().__init__(
            total_images=total_images,
            total_annotations=total_annotations,
            annotated_images=annotated_images,
            unannotated_images=unannotated_images,
            average_image_size=average_image_size,
            object_size_distribution=object_size_distribution,
            class_imbalance=class_imbalance,
            average_objects_per_image=average_objects_per_image,
            average_object_density=average_object_density,
            last_calculated=last_calculated,
            class_counts=class_counts,
            extra_data=extra_data
        )
    
    @staticmethod
    def get_example():
        """Retorna um exemplo para documentação"""
        return {
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
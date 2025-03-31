from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.schemas.base import BaseSchema
from microdetect.schemas.hyperparam_search import TrainingMetrics, ResourceUsage

class ClassPerformance(BaseSchema):
    def __init__(self,
                class_id: int,
                class_name: str,
                precision: float,
                recall: float,
                f1_score: float,
                support: int,
                examples_count: int):
        super().__init__(
            class_id=class_id,
            class_name=class_name,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            support=support,
            examples_count=examples_count
        )

class TrainingReportCreate(BaseSchema):
    def __init__(self,
                training_session_id: int,
                model_name: str,
                dataset_id: int,
                metrics_history: List[TrainingMetrics],
                confusion_matrix: List[List[int]],
                class_performance: List[ClassPerformance],
                final_metrics: Dict[str, Any],
                resource_usage_avg: ResourceUsage,
                resource_usage_max: ResourceUsage,
                hyperparameters: Dict[str, Any],
                train_images_count: int,
                val_images_count: int,
                test_images_count: int,
                training_time_seconds: int,
                model_size_mb: float):
        super().__init__(
            training_session_id=training_session_id,
            model_name=model_name,
            dataset_id=dataset_id,
            metrics_history=metrics_history,
            confusion_matrix=confusion_matrix,
            class_performance=class_performance,
            final_metrics=final_metrics,
            resource_usage_avg=resource_usage_avg,
            resource_usage_max=resource_usage_max,
            hyperparameters=hyperparameters,
            train_images_count=train_images_count,
            val_images_count=val_images_count,
            test_images_count=test_images_count,
            training_time_seconds=training_time_seconds,
            model_size_mb=model_size_mb
        )

class TrainingReportResponse(TrainingReportCreate):
    def __init__(self,
                id: int,
                created_at: datetime,
                **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.created_at = created_at
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto ORM para este schema."""
        # Construir objetos aninhados
        metrics_history = []
        for metric in getattr(obj, 'metrics_history', []):
            if 'resources' in metric and metric['resources']:
                resources = ResourceUsage(**metric['resources'])
                del metric['resources']
                metrics_history.append(TrainingMetrics(**metric, resources=resources))
            else:
                metrics_history.append(TrainingMetrics(**metric))
        
        class_performance = []
        for perf in getattr(obj, 'class_performance', []):
            class_performance.append(ClassPerformance(**perf))
        
        resource_usage_avg = ResourceUsage(**getattr(obj, 'resource_usage_avg', {}))
        resource_usage_max = ResourceUsage(**getattr(obj, 'resource_usage_max', {}))
        
        # Construir o relatório completo
        return cls(
            id=obj.id,
            created_at=obj.created_at,
            training_session_id=obj.training_session_id,
            model_name=obj.model_name,
            dataset_id=obj.dataset_id,
            metrics_history=metrics_history,
            confusion_matrix=getattr(obj, 'confusion_matrix', []),
            class_performance=class_performance,
            final_metrics=getattr(obj, 'final_metrics', {}),
            resource_usage_avg=resource_usage_avg,
            resource_usage_max=resource_usage_max,
            hyperparameters=getattr(obj, 'hyperparameters', {}),
            train_images_count=getattr(obj, 'train_images_count', 0),
            val_images_count=getattr(obj, 'val_images_count', 0),
            test_images_count=getattr(obj, 'test_images_count', 0),
            training_time_seconds=getattr(obj, 'training_time_seconds', 0),
            model_size_mb=getattr(obj, 'model_size_mb', 0)
        ) 
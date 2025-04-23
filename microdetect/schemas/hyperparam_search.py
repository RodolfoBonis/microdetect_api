from datetime import datetime
from typing import Optional, Dict, Any, List
from microdetect.schemas.base import BaseSchema

class HyperparamSearchBase(BaseSchema):
    def __init__(self,
                name: str,
                dataset_id: int,
                search_space: Dict[str, Any],
                iterations: int = 5,
                description: Optional[str] = None):
        super().__init__(
            name=name,
            dataset_id=dataset_id,
            search_space=search_space,
            iterations=iterations,
            description=description
        )

class HyperparamSearchCreate(HyperparamSearchBase):
    """Classe para criação de busca de hiperparâmetros."""
    pass

class HyperparamSearchUpdate(BaseSchema):
    def __init__(self,
                status: Optional[str] = None,
                best_params: Optional[Dict[str, Any]] = None,
                best_metrics: Optional[Dict[str, Any]] = None,
                trials_data: Optional[List[Dict[str, Any]]] = None,
                started_at: Optional[datetime] = None,
                completed_at: Optional[datetime] = None):
        super().__init__(
            status=status,
            best_params=best_params,
            best_metrics=best_metrics,
            trials_data=trials_data,
            started_at=started_at,
            completed_at=completed_at
        )

class HyperparamSearchResponse(HyperparamSearchBase):
    def __init__(self,
                id: int,
                status: str,
                name: str,
                dataset_id: int,
                search_space: Dict[str, Any],
                iterations: int = 5,
                description: Optional[str] = None,
                best_params: Optional[Dict[str, Any]] = None,
                best_metrics: Optional[Dict[str, Any]] = None,
                trials_data: Optional[List[Dict[str, Any]]] = None,
                created_at: datetime = None,
                updated_at: datetime = None,
                started_at: Optional[datetime] = None,
                completed_at: Optional[datetime] = None,
                training_session_id: Optional[int] = None,
                current_iteration: Optional[int] = None,
                iterations_completed: Optional[int] = None,
                current_params: Optional[Dict[str, Any]] = None,
                progress: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=name,
            dataset_id=dataset_id,
            search_space=search_space,
            iterations=iterations,
            description=description
        )
        self.id = id
        self.status = status
        self.best_params = best_params if best_params is not None else {}
        self.best_metrics = best_metrics if best_metrics is not None else {}
        self.trials_data = trials_data if trials_data is not None else []
        self.created_at = created_at
        self.updated_at = updated_at
        self.started_at = started_at
        self.completed_at = completed_at
        self.training_session_id = training_session_id
        
        self.current_iteration = current_iteration
        self.iterations_completed = iterations_completed
        self.current_params = current_params or {}
        self.progress = progress or {}
    
    @classmethod
    def from_orm(cls, obj):
        """Converte um objeto HyperparamSearch para este schema."""
        return cls(
            id=obj.id,
            name=obj.name,
            description=obj.description,
            dataset_id=obj.dataset_id,
            search_space=obj.search_space,
            iterations=obj.iterations,
            status=obj.status,
            best_params=obj.best_params,
            best_metrics=obj.best_metrics,
            trials_data=obj.trials_data,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            started_at=obj.started_at,
            completed_at=obj.completed_at,
            training_session_id=obj.training_session_id,
            current_iteration=len(obj.trials_data),
            iterations_completed=len(obj.trials_data),
            current_params=obj.best_params,
            progress={
                "status": obj.status,
                "trials": obj.trials_data,
                "best_params": obj.best_params,
                "best_metrics": obj.best_metrics,
                "current_iteration": len(obj.trials_data),
                "iterations_completed": len(obj.trials_data),
                "total_iterations": obj.iterations
            }
        )

class HyperparamTrialCreate(BaseSchema):
    def __init__(self,
                hyperparam_search_id: int,
                params: Dict[str, Any],
                trial_number: int):
        super().__init__(
            hyperparam_search_id=hyperparam_search_id,
            params=params,
            trial_number=trial_number
        )

class HyperparamTrialUpdate(BaseSchema):
    def __init__(self,
                metrics: Dict[str, Any],
                completed: bool = True):
        super().__init__(
            metrics=metrics,
            completed=completed
        )

class ResourceUsage(BaseSchema):
    def __init__(self,
                cpu_percent: float,
                memory_percent: float,
                gpu_percent: Optional[float] = None,
                gpu_memory_percent: Optional[float] = None):
        super().__init__(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent
        )

class TrainingMetrics(BaseSchema):
    def __init__(self,
                epoch: int,
                loss: float,
                val_loss: Optional[float] = None,
                map50: Optional[float] = None,
                map: Optional[float] = None,
                precision: Optional[float] = None,
                recall: Optional[float] = None,
                resources: Optional[ResourceUsage] = None):
        super().__init__(
            epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            map50=map50,
            map=map,
            precision=precision,
            recall=recall,
            resources=resources
        )

class TrainingProgress(BaseSchema):
    def __init__(self,
                current_epoch: int,
                total_epochs: int,
                metrics: TrainingMetrics,
                eta_seconds: Optional[int] = None):
        super().__init__(
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            metrics=metrics,
            eta_seconds=eta_seconds
        ) 
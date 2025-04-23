import os
import json
import logging
from celery import Task
from microdetect.core.celery_app import celery_app
from microdetect.services.yolo_service import YOLOService
from microdetect.models.hyperparam_search import HyperparamSearch
from microdetect.database.database import SessionLocal, get_db
from microdetect.core.hyperparam_core import (
    prepare_hyperparam_directory,
    prepare_hyperparam_config,
    HyperparameterOptimizer
)
from datetime import datetime
from microdetect.core.config import settings

logger = logging.getLogger(__name__)

class HyperparamTask(Task):
    _yolo_service = None
    
    @property
    def yolo_service(self):
        if self._yolo_service is None:
            self._yolo_service = YOLOService()
        return self._yolo_service

@celery_app.task(bind=True, base=HyperparamTask)
def run_hyperparameter_search(self, search_id: int):
    """
    Task Celery para executar busca de hiperparâmetros
    """
    db = SessionLocal()
    try:
        search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
        
        if not search:
            raise ValueError(f"Busca de hiperparâmetros {search_id} não encontrada")
            
        # Atualizar status
        search.status = "running"
        db.commit()
        
        # Preparar diretório
        train_dir = prepare_hyperparam_directory(search, settings.TRAINING_DIR)
        
        # Configurar busca
        config = prepare_hyperparam_config(search, train_dir, db)
        
        # Criar otimizador
        optimizer = HyperparameterOptimizer(search.search_space)
        
        best_metrics = None
        best_params = None
        trials_data = []
        
        # Executar trials
        for trial in range(search.iterations):
            # Obter próximos parâmetros para testar
            params = optimizer.suggest_parameters()
            
            # Treinar modelo com esses parâmetros
            metrics = self.yolo_service.train(
                model_path=config["model_path"],
                data_yaml=config["data_yaml"],
                **params
            )
            
            # Atualizar otimizador com resultados
            optimizer.update_results(params, metrics)
            
            # Atualizar dados do trial
            trial_data = {
                "trial": trial + 1,
                "params": params,
                "metrics": metrics
            }
            trials_data.append(trial_data)
            
            # Atualizar melhor resultado
            if best_metrics is None or metrics[config["metric"]] > best_metrics[config["metric"]]:
                best_metrics = metrics
                best_params = params
            
            # Atualizar busca no banco
            search.trials_data = trials_data
            search.best_params = best_params
            search.best_metrics = best_metrics
            db.commit()
        
        # Atualizar status final
        search.status = "completed"
        search.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "status": "success",
            "search_id": search_id,
            "best_params": best_params,
            "best_metrics": best_metrics
        }
        
    except Exception as e:
        logger.error(f"Erro durante otimização: {str(e)}")
        if search:
            search.status = "failed"
            db.commit()
        raise
        
    finally:
        db.close() 
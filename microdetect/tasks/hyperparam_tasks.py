import os
import json
import logging
from celery import Task
from microdetect.core.celery_app import celery_app
from microdetect.services.yolo_service import YOLOService
from microdetect.models.training_session import TrainingSession
from microdetect.core.database import SessionLocal
from microdetect.core.hyperparam_core import (
    prepare_hyperparam_directory,
    prepare_hyperparam_config,
    update_hyperparam_status,
    HyperparameterOptimizer
)

logger = logging.getLogger(__name__)

class HyperparamTask(Task):
    _yolo_service = None
    
    @property
    def yolo_service(self):
        if self._yolo_service is None:
            self._yolo_service = YOLOService()
        return self._yolo_service

@celery_app.task(bind=True, base=HyperparamTask)
def run_hyperparameter_search(self, session_id: int):
    """
    Task Celery para executar busca de hiperparâmetros
    """
    db = SessionLocal()
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise ValueError(f"Sessão de treinamento {session_id} não encontrada")
            
        # Atualizar status
        update_hyperparam_status(session, "running", db=db)
        
        # Preparar diretório
        train_dir = prepare_hyperparam_directory(session, settings.TRAINING_DIR)
        
        # Configurar busca
        config = prepare_hyperparam_config(session, train_dir, db)
        
        # Criar otimizador
        optimizer = HyperparameterOptimizer(config["search_space"])
        
        best_metrics = None
        best_params = None
        
        # Executar trials
        for trial in range(config["max_trials"]):
            # Obter próximos parâmetros para testar
            params = optimizer.suggest_parameters()
            
            # Treinar modelo com esses parâmetros
            metrics = self.yolo_service.train(
                model_path=session.model_path,
                data_yaml=config["data_yaml"],
                **params
            )
            
            # Atualizar otimizador com resultados
            optimizer.update_results(params, metrics)
            
            # Atualizar melhor resultado
            if best_metrics is None or metrics[config["metric"]] > best_metrics[config["metric"]]:
                best_metrics = metrics
                best_params = params
        
        # Atualizar status final
        session.hyperparameters["best_params"] = best_params
        session.metrics = best_metrics
        update_hyperparam_status(session, "completed", db=db)
        
        return {
            "status": "success",
            "session_id": session_id,
            "best_params": best_params,
            "best_metrics": best_metrics
        }
        
    except Exception as e:
        logger.error(f"Erro durante otimização: {str(e)}")
        if session:
            update_hyperparam_status(session, "failed", error_message=str(e), db=db)
        raise
        
    finally:
        db.close() 
import os
import json
import logging
from celery import Task
from microdetect.core.celery_app import celery_app
from microdetect.services.hyperparam_service import HyperparameterOptimizer
from microdetect.services.yolo_service import YOLOService
from microdetect.models.hyperparameter_search import HyperparameterSearch
from microdetect.core.database import SessionLocal

logger = logging.getLogger(__name__)

class HyperparamTask(Task):
    _optimizer = None
    _yolo_service = None
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = HyperparameterOptimizer()
        return self._optimizer
    
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
    try:
        db = SessionLocal()
        search = db.query(HyperparameterSearch).filter(HyperparameterSearch.id == search_id).first()
        
        if not search:
            raise ValueError(f"Busca de hiperparâmetros {search_id} não encontrada")
            
        # Atualizar status
        search.status = "running"
        db.commit()
        
        # Preparar diretório
        search_dir = self.optimizer.prepare_search_directory(search)
        
        # Configurar busca
        config = self.optimizer.prepare_search_config(search, search_dir)
        
        # Executar iterações
        for iteration in range(search.max_iterations):
            # Gerar parâmetros
            params = self.optimizer.generate_parameters()
            
            # Treinar com parâmetros
            metrics = self.yolo_service.train(
                model_path=search.model_path,
                data_yaml=config["data_yaml"],
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                img_size=params["img_size"],
                device=config["device"]
            )
            
            # Atualizar melhor resultado
            self.optimizer.update_best_parameters(params, metrics)
            
            # Salvar progresso
            progress = {
                "iteration": iteration + 1,
                "parameters": params,
                "metrics": metrics,
                "best_parameters": self.optimizer.get_best_parameters()
            }
            
            with open(os.path.join(search_dir, "progress.json"), "w") as f:
                json.dump(progress, f)
        
        # Finalizar busca
        search.status = "completed"
        search.best_parameters = self.optimizer.get_best_parameters()
        db.commit()
        
        return {
            "status": "success",
            "search_id": search_id,
            "best_parameters": self.optimizer.get_best_parameters()
        }
        
    except Exception as e:
        logger.error(f"Erro durante busca de hiperparâmetros: {str(e)}")
        if search:
            search.status = "failed"
            search.error_message = str(e)
            db.commit()
        raise
        
    finally:
        db.close() 
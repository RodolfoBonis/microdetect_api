import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Classe base para otimização de hiperparâmetros.
    """
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Sugere um conjunto de hiperparâmetros para testar.
        """
        raise NotImplementedError()
        
    def update_results(self, parameters: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Atualiza o otimizador com os resultados de um teste.
        """
        raise NotImplementedError()

def prepare_hyperparam_directory(session: TrainingSession, base_dir: Path) -> Path:
    """
    Prepara o diretório para otimização de hiperparâmetros.
    """
    session_dir = base_dir / f"hyperparam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(exist_ok=True)
    return session_dir

def prepare_hyperparam_config(session: TrainingSession, train_dir: Path, db: Session) -> Dict[str, Any]:
    """
    Prepara a configuração para otimização de hiperparâmetros.
    """
    dataset = db.query(Dataset).get(session.dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {session.dataset_id} não encontrado")
        
    config = {
        "data_yaml": str(train_dir / "data.yaml"),
        "search_space": session.hyperparameters.get("search_space", {}),
        "max_trials": session.hyperparameters.get("max_trials", 10),
        "metric": session.hyperparameters.get("metric", "map"),
        "direction": session.hyperparameters.get("direction", "maximize")
    }
    
    return config

def update_hyperparam_status(session: TrainingSession, status: str, error_message: str = None, db: Session = None):
    """
    Atualiza o status de uma sessão de otimização de hiperparâmetros.
    """
    session.status = status
    if error_message:
        session.error_message = error_message
    if db:
        db.commit() 
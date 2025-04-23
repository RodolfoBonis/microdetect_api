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
from microdetect.models.hyperparam_search import HyperparamSearch

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

def prepare_hyperparam_config(search: HyperparamSearch, train_dir: str, db: Session) -> Dict[str, Any]:
    """
    Prepara a configuração para otimização de hiperparâmetros.
    """
    try:
        # Verificar se o diretório existe
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
            
        # Criar arquivo de configuração
        config = {
            "dataset_id": search.dataset_id,
            "search_space": search.search_space,  # Usar search_space ao invés de hyperparameters
            "max_trials": search.iterations,
            "train_dir": train_dir,
            "model_type": "yolov8",  # TODO: Tornar configurável
            "model_version": "n",     # TODO: Tornar configurável
            "device": "cpu" if not settings.USE_CUDA else "cuda"
        }
        
        # Salvar configuração
        config_path = os.path.join(train_dir, "hyperparam_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
        return config
        
    except Exception as e:
        logger.error(f"Erro ao preparar configuração: {str(e)}")
        raise

def update_hyperparam_status(session: TrainingSession, status: str, error_message: str = None, db: Session = None):
    """
    Atualiza o status de uma sessão de otimização de hiperparâmetros.
    """
    session.status = status
    if error_message:
        session.error_message = error_message
    if db:
        db.commit() 
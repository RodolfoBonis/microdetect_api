import os
import json
import logging
import random
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

class RandomSearchOptimizer(HyperparameterOptimizer):
    """
    Implementação de busca aleatória de hiperparâmetros.
    """
    def __init__(self, search_space: Dict[str, Any]):
        super().__init__(search_space)
        self.best_metrics = None
        self.best_params = None
        
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Sugere parâmetros aleatórios dentro do espaço de busca.
        """
        params = {}
        for param_name, param_space in self.search_space.items():
            # Converter learning_rate para lr0
            actual_param_name = "lr0" if param_name == "learning_rate" else param_name
            
            if isinstance(param_space, dict):
                if "min" in param_space and "max" in param_space:
                    # Parâmetro numérico com intervalo
                    min_val = float(param_space["min"])
                    max_val = float(param_space["max"])
                    if param_name in ["batch_size", "epochs"]:
                        # Parâmetros inteiros
                        params[actual_param_name] = random.randint(int(min_val), int(max_val))
                    else:
                        # Parâmetros float
                        params[actual_param_name] = random.uniform(min_val, max_val)
            elif isinstance(param_space, list):
                # Escolha aleatória de uma opção
                params[actual_param_name] = random.choice(param_space)
            else:
                # Valor fixo
                params[actual_param_name] = param_space
                
        return params
        
    def update_results(self, parameters: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Atualiza o melhor resultado encontrado.
        """
        # Usar mAP como métrica principal (média de mAP50 e mAP)
        current_map = (metrics.get("best_map50", 0.0) + metrics.get("best_map", 0.0)) / 2
        
        if self.best_metrics is None or current_map > (self.best_metrics.get("best_map50", 0.0) + self.best_metrics.get("best_map", 0.0)) / 2:
            self.best_metrics = metrics
            self.best_params = parameters

def prepare_hyperparam_directory(session: TrainingSession, base_dir: str | Path) -> Path:
    """
    Prepara o diretório para otimização de hiperparâmetros.
    """
    base_path = Path(base_dir) if isinstance(base_dir, str) else base_dir
    base_path.mkdir(parents=True, exist_ok=True)  # Garantir que o diretório base exista
    session_dir = base_path / f"hyperparam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(exist_ok=True)
    return session_dir

def prepare_hyperparam_config(search: HyperparamSearch, train_dir: str | Path, db: Session) -> Dict[str, Any]:
    """
    Prepara a configuração para otimização de hiperparâmetros.
    """
    try:
        # Converter train_dir para Path se for string
        train_dir_path = Path(train_dir) if isinstance(train_dir, str) else train_dir
        
        # Verificar se o diretório existe
        if not train_dir_path.exists():
            train_dir_path.mkdir(parents=True, exist_ok=True)
            
        # Criar arquivo de configuração
        config = {
            "dataset_id": search.dataset_id,
            "search_space": search.search_space,  # Usar search_space ao invés de hyperparameters
            "max_trials": search.iterations,
            "train_dir": str(train_dir_path),  # Converter Path para string
            "model_type": "yolov8",  # TODO: Tornar configurável
            "model_version": "n",     # TODO: Tornar configurável
            "device": "cpu" if not settings.USE_CUDA else "cuda"
        }
        
        # Salvar configuração
        config_path = train_dir_path / "hyperparam_config.json"
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
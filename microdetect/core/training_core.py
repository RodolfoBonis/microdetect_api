import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

def prepare_training_directory(session: TrainingSession, training_dir: Path) -> Path:
    """
    Prepara o diretório de treinamento para uma sessão.
    """
    session_dir = training_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(exist_ok=True)
    return session_dir

def prepare_training_config(session: TrainingSession, train_dir: Path, db: Session) -> Dict[str, Any]:
    """
    Prepara a configuração de treinamento.
    """
    dataset = db.query(Dataset).get(session.dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {session.dataset_id} não encontrado")
        
    config = {
        "data_yaml": str(train_dir / "data.yaml"),
        "epochs": session.hyperparameters.get("epochs", 100),
        "batch_size": session.hyperparameters.get("batch_size", 16),
        "img_size": session.hyperparameters.get("img_size", 640),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return config

def update_training_status(session: TrainingSession, status: str, error_message: str = None, db: Session = None):
    """
    Atualiza o status de uma sessão de treinamento.
    """
    session.status = status
    if error_message:
        session.error_message = error_message
    if db:
        db.commit() 
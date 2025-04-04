import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService
from sqlalchemy.orm import Session
from microdetect.database.database import get_db
import shutil
import asyncio
import logging
import torch
from microdetect.core.websocket_manager import WebSocketManager
from microdetect.core.hyperparam_core import (
    prepare_hyperparam_directory,
    prepare_hyperparam_config,
    update_hyperparam_status,
    HyperparameterOptimizer
)

logger = logging.getLogger(__name__)

class HyperparamService:
    def __init__(self):
        self.training_dir = settings.TRAINING_DIR
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_service = YOLOService()
        self._db = next(get_db())
        self.websocket_manager = WebSocketManager()

    async def create_hyperparam_session(
        self,
        dataset_id: int,
        model_type: str,
        model_version: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        search_space: Optional[Dict[str, Any]] = None,
        max_trials: int = 10
    ) -> TrainingSession:
        """
        Cria uma nova sessão de otimização de hiperparâmetros.
        """
        # Verificar dataset
        dataset = self._db.query(Dataset).get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Criar diretório da sessão
        session_dir = prepare_hyperparam_directory(None, self.training_dir)
        
        # Criar registro no banco
        session = TrainingSession(
            name=name or f"Otimização {dataset.name}",
            description=description,
            dataset_id=dataset_id,
            model_type=model_type,
            model_version=model_version,
            hyperparameters={
                "search_space": search_space or {},
                "max_trials": max_trials
            },
            status="pending",
            metrics={},
            log_file=str(session_dir / "hyperparam.log")
        )
        
        # Adicionar e salvar no banco
        self._db.add(session)
        self._db.commit()
        self._db.refresh(session)
        
        return session

    async def start_hyperparam_search(self, session_id: int) -> TrainingSession:
        """
        Inicia a busca de hiperparâmetros usando Celery.
        """
        session = self._db.query(TrainingSession).get(session_id)
        if not session:
            raise ValueError(f"Sessão {session_id} não encontrada")
        
        # Atualizar status
        update_hyperparam_status(session, "running", db=self._db)
        session.started_at = datetime.utcnow()
        self._db.commit()
        
        # Importar aqui para evitar importação circular
        from microdetect.tasks.hyperparam_tasks import run_hyperparameter_search
        
        # Iniciar task Celery
        task = run_hyperparameter_search.delay(session_id)
        
        # Iniciar monitoramento via WebSocket
        asyncio.create_task(self._monitor_search_progress(session_id, task.id))
        
        return session

    async def _monitor_search_progress(self, session_id: int, task_id: str):
        """
        Monitora o progresso da busca de hiperparâmetros.
        """
        try:
            while True:
                # Obter status da task
                from microdetect.tasks.hyperparam_tasks import run_hyperparameter_search
                task = run_hyperparameter_search.AsyncResult(task_id)
                
                if task.ready():
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Erro ao monitorar progresso: {str(e)}")
            
    def __del__(self):
        """
        Fechar a sessão do banco quando o serviço for destruído
        """
        if hasattr(self, '_db'):
            self._db.close() 
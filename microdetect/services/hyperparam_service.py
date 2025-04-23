import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession
from microdetect.models.hyperparam_search import HyperparamSearch
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
from microdetect.tasks.hyperparam_tasks import run_hyperparameter_search

logger = logging.getLogger(__name__)

class HyperparamService:
    def __init__(self):
        self.training_dir = Path(settings.TRAINING_DIR)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda:0" if settings.USE_CUDA else "cpu"
        logger.info(f"CUDA available in HyperparamService: {settings.USE_CUDA}")

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
    ) -> HyperparamSearch:
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
        search = HyperparamSearch(
            name=name or f"Otimização {dataset.name}",
            description=description,
            dataset_id=dataset_id,
            status="pending",
            search_space=search_space or {},
            iterations=max_trials,
            trials_data=[],
            best_params={},
            best_metrics={}
        )
        
        # Adicionar e salvar no banco
        self._db.add(search)
        self._db.commit()
        self._db.refresh(search)
        
        return search

    async def start_hyperparam_search(self, search_id: int) -> HyperparamSearch:
        """
        Inicia a busca de hiperparâmetros usando Celery.
        """
        search = self._db.query(HyperparamSearch).get(search_id)
        if not search:
            raise ValueError(f"Busca {search_id} não encontrada")
        
        # Atualizar status
        search.status = "running"
        search.started_at = datetime.utcnow()
        self._db.commit()
        
        # Iniciar task Celery
        task = run_hyperparameter_search.delay(search_id)
        
        # Iniciar monitoramento via WebSocket
        asyncio.create_task(self._monitor_search_progress(search_id, task.id))
        
        return search

    async def _monitor_search_progress(self, search_id: int, task_id: str):
        """
        Monitora o progresso da busca de hiperparâmetros.
        """
        try:
            while True:
                # Obter status da task
                task = run_hyperparameter_search.AsyncResult(task_id)
                
                if task.ready():
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Erro ao monitorar progresso: {str(e)}")

    async def list_searches(
        self,
        dataset_id: Optional[int] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        db: Optional[Session] = None
    ) -> List[HyperparamSearch]:
        """
        Lista as buscas de hiperparâmetros com filtros opcionais.
        
        Args:
            dataset_id: ID do dataset para filtrar
            status: Status da busca para filtrar
            skip: Número de registros para pular
            limit: Número máximo de registros para retornar
            db: Sessão do banco de dados (opcional)
            
        Returns:
            Lista de buscas de hiperparâmetros
        """
        query = self._db.query(HyperparamSearch)
        
        # Aplicar filtros
        if dataset_id is not None:
            query = query.filter(HyperparamSearch.dataset_id == dataset_id)
        if status is not None:
            query = query.filter(HyperparamSearch.status == status)
            
        # Ordenar por data de criação (mais recentes primeiro)
        query = query.order_by(HyperparamSearch.created_at.desc())
        
        # Aplicar paginação
        query = query.offset(skip).limit(limit)
        
        return query.all()
    
    async def get_search(self, search_id: int, db: Optional[Session] = None) -> Optional[HyperparamSearch]:
        """
        Obtém uma busca de hiperparâmetros pelo ID.
        
        Args:
            search_id: ID da busca
            db: Sessão do banco de dados (opcional)
            
        Returns:
            Busca de hiperparâmetros ou None se não encontrada
        """
        return self._db.query(HyperparamSearch).get(search_id)
    
    async def delete_search(self, search_id: int, db: Optional[Session] = None) -> bool:
        """
        Remove uma busca de hiperparâmetros.
        
        Args:
            search_id: ID da busca
            db: Sessão do banco de dados (opcional)
            
        Returns:
            True se a busca foi removida, False caso contrário
        """
        search = self._db.query(HyperparamSearch).get(search_id)
        if not search:
            return False
            
        # Remover do banco
        self._db.delete(search)
        self._db.commit()
        
        return True
    
    def get_progress(self, search_id: int) -> Dict[str, Any]:
        """
        Obtém o progresso atual de uma busca de hiperparâmetros.
        
        Args:
            search_id: ID da busca
            
        Returns:
            Dicionário com informações de progresso
        """
        try:
            search = self._db.query(HyperparamSearch).get(search_id)
            if not search:
                logger.warning(f"Busca de hiperparâmetros {search_id} não encontrada")
                return {}
                
            # Garantir que os campos JSON não sejam None
            trials_data = search.trials_data if search.trials_data is not None else []
            best_params = search.best_params if search.best_params is not None else {}
            best_metrics = search.best_metrics if search.best_metrics is not None else {}
            
            return {
                "status": search.status,
                "trials": trials_data,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "current_iteration": len(trials_data),
                "iterations_completed": len(trials_data),
                "total_iterations": search.iterations
            }
        except Exception as e:
            logger.error(f"Erro ao obter progresso da busca {search_id}: {str(e)}")
            return {}
            
    def __del__(self):
        """
        Fechar a sessão do banco quando o serviço for destruído
        """
        if hasattr(self, '_db'):
            self._db.close() 
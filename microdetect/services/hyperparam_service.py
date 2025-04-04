import os
import shutil
import json
import logging
import random
import asyncio
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from microdetect.core.config import settings
from microdetect.models.hyperparam_search import HyperparamSearch, HyperparamSearchStatus
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.services.yolo_service import YOLOService
from microdetect.services.resource_monitor import ResourceMonitor
from microdetect.schemas.hyperparam_search import ResourceUsage, TrainingMetrics
from microdetect.database.database import get_db
from microdetect.services.dataset_service import DatasetService
from microdetect.models.dataset import Dataset
from microdetect.tasks.hyperparam_tasks import run_hyperparameter_search
from microdetect.core.websocket_manager import WebSocketManager

# Configurar logging
logger = logging.getLogger(__name__)

# Verificar se CUDA está disponível
CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(f"CUDA available: {CUDA_AVAILABLE}")

class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros usando estratégias simples."""
    
    def __init__(self, search_space: Dict[str, Any], iterations: int = 5):
        self.search_space = search_space
        self.iterations = iterations
        self.trials = []
        self.best_params = None
        self.best_metric = 0.0  # Valor inicial baixo para maximização
        self.metric_key = "map50"  # Métrica padrão a otimizar
    
    def generate_params(self) -> Dict[str, Any]:
        """Gera um conjunto de hiperparâmetros a partir do espaço de busca."""
        params = {}
        for param_name, param_space in self.search_space.items():
            if isinstance(param_space, list):
                # Lista de valores discretos
                params[param_name] = random.choice(param_space)
            elif isinstance(param_space, dict):
                if "min" in param_space and "max" in param_space:
                    # Faixa numérica
                    min_val = param_space["min"]
                    max_val = param_space["max"]
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Inteiro
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        # Float
                        params[param_name] = min_val + random.random() * (max_val - min_val)
                elif "options" in param_space:
                    # Opções discretas
                    params[param_name] = random.choice(param_space["options"])
            elif isinstance(param_space, tuple) and len(param_space) == 2:
                # Tupla (min, max)
                min_val, max_val = param_space
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = min_val + random.random() * (max_val - min_val)
            else:
                # Valor fixo
                params[param_name] = param_space
        
        return params
    
    def update_best(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Atualiza o melhor conjunto de hiperparâmetros com base nas métricas."""
        if not metrics or self.metric_key not in metrics:
            return
            
        metric_value = metrics[self.metric_key]
        
        # Registrar a tentativa
        self.trials.append({
            "params": params,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Atualizar o melhor se necessário
        if self.best_params is None or metric_value > self.best_metric:
            self.best_params = params
            self.best_metric = metric_value
            logger.info(f"New best params found: {self.metric_key}={metric_value}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Retorna os melhores hiperparâmetros encontrados."""
        return self.best_params or self.generate_params()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Retorna as métricas dos melhores hiperparâmetros."""
        if not self.trials:
            return {}
            
        for trial in self.trials:
            if trial["params"] == self.best_params:
                return trial["metrics"]
                
        return {}
    
    def get_trials_data(self) -> List[Dict[str, Any]]:
        """Retorna os dados de todas as tentativas."""
        return self.trials


class HyperparamService:
    """Serviço para busca de hiperparâmetros."""
    
    def __init__(self):
        self.yolo_service = YOLOService()
        self.resource_monitor = ResourceMonitor()
        self._db = next(get_db())  # Obter uma sessão do banco para usar nos métodos
        self.websocket_manager = WebSocketManager()
    
    def __del__(self):
        """Fecha a sessão do banco quando o serviço for destruído."""
        try:
            self._db.close()
        except:
            pass
    
    async def create_search(self, search_data: Dict[str, Any], db: Session) -> HyperparamSearch:
        """
        Cria uma nova busca de hiperparâmetros.
        
        Args:
            search_data: Dados da busca
            db: Sessão do banco de dados
            
        Returns:
            Objeto de busca criado
        """
        search = HyperparamSearch(**search_data)
        db.add(search)
        db.commit()
        db.refresh(search)
        return search
    
    async def start_search(self, search_id: int, db: Session) -> None:
        """
        Inicia a busca de hiperparâmetros usando Celery.
        
        Args:
            search_id: ID da busca
            db: Sessão do banco de dados
        """
        search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
        if not search:
            raise ValueError(f"Search {search_id} not found")
            
        if search.status != HyperparamSearchStatus.PENDING:
            logger.warning(f"Search {search_id} is already {search.status}")
            return
            
        # Atualizar status
        search.status = HyperparamSearchStatus.RUNNING
        search.started_at = datetime.utcnow()
        db.commit()
        
        # Iniciar task Celery
        task = run_hyperparameter_search.delay(search_id)
        
        # Iniciar monitoramento via WebSocket
        asyncio.create_task(self._monitor_search_progress(search_id, task.id))
    
    async def _monitor_search_progress(self, search_id: int, task_id: str):
        """
        Monitora o progresso da busca e envia atualizações via WebSocket.
        """
        try:
            while True:
                # Obter status da task
                task = run_hyperparameter_search.AsyncResult(task_id)
                
                if task.ready():
                    # Busca concluída
                    if task.successful():
                        result = task.get()
                        await self.websocket_manager.broadcast_json(
                            f"hyperparam_{search_id}",
                            {
                                "status": "completed",
                                "best_params": result.get("best_params", {}),
                                "best_metrics": result.get("best_metrics", {}),
                                "message": "Busca de hiperparâmetros concluída com sucesso"
                            }
                        )
                    else:
                        error = str(task.result)
                        await self.websocket_manager.broadcast_json(
                            f"hyperparam_{search_id}",
                            {
                                "status": "failed",
                                "error": error,
                                "message": "Erro durante a busca de hiperparâmetros"
                            }
                        )
                    break
                
                # Obter progresso atual
                search = self._db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
                if search and search.trials_data:
                    await self.websocket_manager.broadcast_json(
                        f"hyperparam_{search_id}",
                        {
                            "status": "running",
                            "current_iteration": len(search.trials_data),
                            "total_iterations": search.iterations,
                            "best_params": search.best_params,
                            "best_metrics": search.best_metrics,
                            "trials": search.trials_data
                        }
                    )
                
                await asyncio.sleep(1)  # Atualizar a cada segundo
                
        except Exception as e:
            logger.error(f"Erro ao monitorar progresso da busca: {e}")
            await self.websocket_manager.broadcast_json(
                f"hyperparam_{search_id}",
                {
                    "status": "error",
                    "error": str(e),
                    "message": "Erro ao monitorar progresso"
                }
            )
    
    async def get_search(self, search_id: int, db: Session) -> Optional[HyperparamSearch]:
        """
        Obtém uma busca por ID.
        
        Args:
            search_id: ID da busca
            db: Sessão do banco de dados
            
        Returns:
            Objeto de busca ou None se não encontrado
        """
        return db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
    
    async def list_searches(
        self,
        dataset_id: Optional[int] = None,
        status: Optional[HyperparamSearchStatus] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = None
    ) -> List[HyperparamSearch]:
        """
        Lista buscas com filtros opcionais.
        
        Args:
            dataset_id: Filtrar por dataset
            status: Filtrar por status
            skip: Número de itens para pular
            limit: Número máximo de itens a retornar
            db: Sessão do banco de dados
            
        Returns:
            Lista de buscas
        """
        # Se não for fornecida sessão do banco, usar a interna
        if db is None:
            db = self._db
            
        query = db.query(HyperparamSearch)
        
        if dataset_id:
            query = query.filter(HyperparamSearch.dataset_id == dataset_id)
            
        if status:
            query = query.filter(HyperparamSearch.status == status)
            
        return query.offset(skip).limit(limit).all()
    
    async def delete_search(self, search_id: int, db: Session) -> bool:
        """
        Exclui uma busca por ID.
        
        Args:
            search_id: ID da busca
            db: Sessão do banco de dados
            
        Returns:
            True se excluída com sucesso
        """
        search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
        if not search:
            return False
            
        db.delete(search)
        db.commit()
        return True 
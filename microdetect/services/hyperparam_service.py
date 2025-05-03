from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.hyperparam_search import HyperparamSearch
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService
from sqlalchemy.orm import Session
from microdetect.database.database import get_db
import asyncio
import logging
from microdetect.core.websocket_utils import send_hyperparam_update
from microdetect.core.hyperparam_core import (
    prepare_hyperparam_directory
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
        Inicia uma busca de hiperparâmetros usando Celery.
        
        Args:
            search_id: ID da busca de hiperparâmetros
            
        Returns:
            Objeto HyperparamSearch atualizado
        """
        # Obter busca
        search = self._db.query(HyperparamSearch).get(search_id)
        if not search:
            raise ValueError(f"Busca de hiperparâmetros {search_id} não encontrada")
            
        # Atualizar status
        search.status = "running"
        search.started_at = datetime.utcnow()
        self._db.commit()
        
        # Obter dataset para preparar data_yaml
        dataset = self._db.query(Dataset).get(search.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {search.dataset_id} não encontrado")
        
        # Extrair informações do modelo a partir do search_space ou usar valores padrão
        model_type = "yolov8"  # Valor padrão
        model_version = "n"    # Valor padrão
        
        # Extrair parâmetros e configurações
        param_space = search.search_space or {}
        n_trials = search.iterations or 10
        search_algorithm = "random"  # Valor padrão
        objective_metric = "map"    # Valor padrão
        
        # Iniciar tarefa Celery com todos os parâmetros necessários
        task = run_hyperparameter_search.delay(
            search_id=search_id,
            dataset_id=search.dataset_id,
            param_space=param_space,
            model_type=model_type,
            model_version=model_version,
            n_trials=n_trials,
            search_algorithm=search_algorithm,
            objective_metric=objective_metric
        )
        
        # Iniciar monitoramento via WebSocket
        asyncio.create_task(self._monitor_search_progress(search_id, task.id))
        
        return search
    
    async def _monitor_search_progress(self, search_id: int, task_id: str):
        """
        Monitora o progresso da busca de hiperparâmetros.
        """
        try:
            last_iteration = -1
            last_reported_epoch = -1
            
            logger.info(f"Iniciando monitoramento da busca {search_id}, task_id={task_id}")
            
            while True:
                # Obter status da task
                task = run_hyperparameter_search.AsyncResult(task_id)
                
                # Obter dados atualizados da busca
                search = self._db.query(HyperparamSearch).get(search_id)
                if not search:
                    logger.error(f"Busca {search_id} não encontrada")
                    break
                
                # Obter progresso atual
                progress = self.get_progress(search_id)
                current_trial = len(search.trials_data or [])
                
                # Log de status da tarefa
                if task.state != 'PENDING':
                    logger.debug(f"Task state: {task.state}, info: {task.info}")
                
                # Verificar se houve mudança no trial atual
                if current_trial > last_iteration:
                    last_iteration = current_trial
                    last_reported_epoch = -1  # Reset do controle de épocas ao começar um novo trial
                    logger.info(f"Novo trial iniciado: {current_trial}/{search.iterations}")
                
                # Verificar progresso da época dentro do trial atual
                if task.info and isinstance(task.info, dict):
                    # Extrair informações de época do info da task
                    current_epoch = task.info.get("epoch", 0)
                    progress_type = task.info.get("progress_type", "")
                    total_epochs = task.info.get("total_epochs", 100)
                    
                    logger.debug(f"Task info: epoch={current_epoch}, tipo={progress_type}")
                    
                    # Sempre enviar atualizações de época
                    if progress_type == "epoch" or progress_type == "epoch_in_trial":
                        last_reported_epoch = current_epoch
                        logger.info(f"Trial {current_trial}: Progresso época: {current_epoch}, tipo={progress_type}")
                        
                        # Debug valores brutos
                        logger.info(f"Valores brutos: current_trial={current_trial}, search.iterations={search.iterations}, current_epoch={current_epoch}, total_epochs={total_epochs}")
                        
                        # Nova fórmula simplificada para percent_complete
                        # Garantir que current_trial seja pelo menos 1 para cálculos
                        trial_idx = max(1, current_trial)  # Usar 1 para o primeiro trial
                        
                        # Calcular progresso baseado no índice do trial (1-based) e na proporção da época atual
                        # Para 5 trials, cada um vale 20% do progresso
                        trial_percent = (trial_idx - 1) * (100 / search.iterations)
                        epoch_percent = (current_epoch / max(1, total_epochs)) * (100 / search.iterations)
                        
                        # Somar os dois componentes do progresso
                        percent_complete = trial_percent + epoch_percent
                        
                        # Garantir que seja pelo menos 1 se estiver em andamento, para feedback visual
                        if percent_complete < 1 and current_epoch > 0:
                            percent_complete = 1
                            
                        # Converter para inteiro mantendo arredondamento correto
                        percent_complete = int(round(percent_complete))
                        
                        # Garantir limites
                        percent_complete = max(0, min(100, percent_complete))
                        
                        logger.info(f"Cálculo detalhado: trial_idx={trial_idx}, trial_percent={trial_percent:.2f}%, epoch_percent={epoch_percent:.2f}%, final={percent_complete}%")
                        
                        # Enviar atualização detalhada via WebSocket
                        await send_hyperparam_update(
                            str(search_id),
                            {
                                "status": search.status,
                                "trials": search.trials_data or [],
                                "best_params": search.best_params or {},
                                "best_metrics": search.best_metrics or {},
                                "current_trial": current_trial,
                                "total_trials": search.iterations,
                                "progress": {
                                    "trial": current_trial,
                                    "total_trials": search.iterations,
                                    "current_epoch": current_epoch,
                                    "total_epochs": total_epochs,
                                    "percent_complete": percent_complete
                                },
                                "current_trial_info": task.info
                            }
                        )
                        logger.info(f"Atualização via WebSocket enviada para a busca {search_id}")
                
                # Se a task terminou, verificar resultado
                if task.ready():
                    if task.successful():
                        result = task.get()
                        logger.info(f"Busca {search_id} concluída com sucesso: {result}")
                        await send_hyperparam_update(
                            str(search_id),
                            {
                                "status": "completed",
                                "best_params": result.get("best_params", {}),
                                "best_metrics": result.get("best_metrics", {}),
                                "message": "Busca concluída com sucesso"
                            }
                        )
                    else:
                        error = str(task.result)
                        logger.error(f"Erro na busca {search_id}: {error}")
                        await send_hyperparam_update(
                            str(search_id),
                            {
                                "status": "failed",
                                "error": error,
                                "message": "Erro durante a busca"
                            }
                        )
                    break
                
                # Atualizar a cada 100ms para mais realtime
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Erro ao monitorar progresso: {str(e)}")
            await send_hyperparam_update(
                str(search_id),
                {
                    "status": "error",
                    "error": str(e),
                    "message": "Erro ao monitorar progresso"
                }
            )

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
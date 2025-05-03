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
        search_space: Optional[Dict[str, Any]] = None
    ) -> HyperparamSearch:
        """
        Cria uma nova sessão de otimização de hiperparâmetros.
        """
        # Verificar dataset
        dataset = self._db.query(Dataset).get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Validar search_space
        if search_space:
            # Validar model_type
            if "model_type" in search_space:
                if not isinstance(search_space["model_type"], str):
                    raise ValueError("model_type deve ser uma string")
                valid_types = ["yolov8", "yolov10"]  # Lista em lowercase para comparação
                model_type = search_space["model_type"].lower()  # Converter para lowercase
                if model_type not in valid_types:
                    raise ValueError(f"Tipo de modelo inválido: {search_space['model_type']}. Deve ser um dos: {[t.upper() for t in valid_types]}")
                # Atualizar o valor no search_space para o formato padrão
                search_space["model_type"] = model_type
            
            # Validar model_size
            if "model_size" in search_space:
                if not isinstance(search_space["model_size"], list):
                    raise ValueError("model_size deve ser um array de strings")
                valid_sizes = ["n", "s", "m", "l", "x"]
                for size in search_space["model_size"]:
                    if size not in valid_sizes:
                        raise ValueError(f"Tamanho de modelo inválido: {size}. Deve ser um dos: {valid_sizes}")
            
            # Validar imgsz
            if "imgsz" in search_space:
                if not isinstance(search_space["imgsz"], list):
                    raise ValueError("imgsz deve ser um array de inteiros")
                for size in search_space["imgsz"]:
                    if not isinstance(size, int) or size < 32 or size > 2048:
                        raise ValueError(f"Tamanho de imagem inválido: {size}. Deve ser um inteiro entre 32 e 2048")
            
            # Validar optimizer
            if "optimizer" in search_space:
                if not isinstance(search_space["optimizer"], list):
                    raise ValueError("optimizer deve ser um array de strings")
                valid_optimizers = ["Adam", "SGD", "AdamW", "RMSProp"]
                for opt in search_space["optimizer"]:
                    if opt not in valid_optimizers:
                        raise ValueError(f"Otimizador inválido: {opt}. Deve ser um dos: {valid_optimizers}")
            
            # Validar device
            if "device" in search_space:
                device = search_space["device"]
                if not isinstance(device, str):
                    raise ValueError("device deve ser uma string")
                valid_devices = ["auto", "cpu", "GPU0", "GPU1"]
                if device not in valid_devices and not device.startswith("GPU"):
                    raise ValueError(f"Dispositivo inválido: {device}. Deve ser um dos: {valid_devices} ou GPU seguido de número")
        
        # Criar diretório da sessão
        session_dir = prepare_hyperparam_directory(None, self.training_dir)
        
        # Criar registro no banco
        search = HyperparamSearch(
            name=name or f"Otimização {dataset.name}",
            description=description,
            dataset_id=dataset_id,
            status="pending",
            search_space=search_space or {},
            iterations=0,  # Será atualizado quando iniciar a busca
            trials_data=[],
            best_params={},
            best_metrics={}
        )
        
        # Adicionar e salvar no banco
        self._db.add(search)
        self._db.commit()
        self._db.refresh(search)
        
        return search

    def _generate_param_combinations(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gera todas as combinações possíveis de parâmetros a partir do search_space.
        """
        from itertools import product
        
        # Separar parâmetros que são arrays (para combinação) dos que são valores únicos
        array_params = {}
        single_params = {}
        
        for key, value in search_space.items():
            if isinstance(value, list):
                array_params[key] = value
            elif isinstance(value, dict) and "min" in value and "max" in value:
                # Para parâmetros com min/max, gerar valores intermediários
                if key == "epochs":
                    # Para epochs, gerar valores inteiros
                    min_val = int(value["min"])
                    max_val = int(value["max"])
                    step = max(1, (max_val - min_val) // 3)  # Gerar 3 valores intermediários
                    array_params[key] = list(range(min_val, max_val + 1, step))
                elif key == "batch_size":
                    # Para batch_size, preferir potências de 2
                    min_val = int(value["min"])
                    max_val = int(value["max"])
                    # Encontrar as potências de 2 dentro do intervalo
                    powers = []
                    power = 1
                    while power <= max_val:
                        if power >= min_val:
                            powers.append(power)
                        power *= 2
                    array_params[key] = powers
                elif key == "learning_rate":
                    # Para learning_rate, gerar valores logarítmicos
                    min_val = float(value["min"])
                    max_val = float(value["max"])
                    # Gerar 3 valores logarítmicos
                    import numpy as np
                    array_params[key] = list(np.logspace(np.log10(min_val), np.log10(max_val), num=3))
            else:
                single_params[key] = value
        
        # Gerar todas as combinações possíveis dos parâmetros em array
        param_names = list(array_params.keys())
        param_values = [array_params[name] for name in param_names]
        combinations = list(product(*param_values))
        
        # Construir a lista final de combinações
        result = []
        for combo in combinations:
            # Criar dicionário com os valores da combinação atual
            param_dict = {name: value for name, value in zip(param_names, combo)}
            
            # Converter batch_size para batch (nome usado pelo YOLO)
            if "batch_size" in param_dict:
                param_dict["batch"] = param_dict.pop("batch_size")
            
            # Converter learning_rate para lr0 (nome usado pelo YOLO)
            if "learning_rate" in param_dict:
                param_dict["lr0"] = param_dict.pop("learning_rate")
            
            # Adicionar os parâmetros únicos
            param_dict.update(single_params)
            result.append(param_dict)
        
        return result

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
        
        # Gerar todas as combinações possíveis de parâmetros
        param_combinations = self._generate_param_combinations(search.search_space)
        
        # Atualizar o número de trials com base nas combinações
        search.iterations = len(param_combinations)
        search.trials_data = []  # Inicializar lista vazia de trials
        self._db.commit()
        
        # Iniciar tarefa Celery com todas as combinações
        task = run_hyperparameter_search.delay(
            search_id=search_id,
            dataset_id=search.dataset_id,
            param_space=param_combinations,
            model_type=search.search_space.get("model_type", "yolov8"),
            model_version=search.search_space.get("model_size", "n"),
            data_yaml_path=None
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
                # Se task.info trouxer current_trial em execução, usar esse valor
                if task.info and isinstance(task.info, dict) and "current_trial" in task.info:
                    current_trial = task.info["current_trial"]
                
                # Log de status da tarefa
                if task.state != 'PENDING':
                    logger.debug(f"Task state: {task.state}, info: {task.info}")
                
                # Heartbeat: envie a cada 10 segundos se não houver progresso novo
                import time
                if not hasattr(self, '_last_ws_heartbeat'):
                    self._last_ws_heartbeat = 0
                now = time.time()
                send_heartbeat = False
                if now - self._last_ws_heartbeat > 10:
                    send_heartbeat = True
                    self._last_ws_heartbeat = now
                
                # Verificar se houve mudança no trial atual
                if current_trial > last_iteration:
                    last_iteration = current_trial
                    last_reported_epoch = -1  # Reset do controle de épocas ao começar um novo trial
                    logger.info(f"Novo trial iniciado: {current_trial}/{search.iterations}")
                    send_heartbeat = True  # Sempre envie update ao mudar de trial
                
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
                        trial_idx = max(1, current_trial)
                        trial_percent = (trial_idx - 1) * (100 / search.iterations)
                        epoch_percent = (current_epoch / max(1, total_epochs)) * (100 / search.iterations)
                        percent_complete = trial_percent + epoch_percent
                        if percent_complete < 1 and current_epoch > 0:
                            percent_complete = 1
                        percent_complete = int(round(percent_complete))
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
                        self._last_ws_heartbeat = now
                    elif send_heartbeat:
                        # Enviar heartbeat mesmo sem progresso novo
                        await send_hyperparam_update(
                            str(search_id),
                            {
                                "status": search.status,
                                "trials": search.trials_data or [],
                                "best_params": search.best_params or {},
                                "best_metrics": search.best_metrics or {},
                                "current_trial": current_trial,
                                "total_trials": search.iterations,
                                "progress": {},
                                "heartbeat": True
                            }
                        )
                        logger.info(f"Heartbeat WebSocket enviado para a busca {search_id}")
                elif send_heartbeat:
                    # Enviar heartbeat mesmo sem progresso novo
                    await send_hyperparam_update(
                        str(search_id),
                        {
                            "status": search.status,
                            "trials": search.trials_data or [],
                            "best_params": search.best_params or {},
                            "best_metrics": search.best_metrics or {},
                            "current_trial": current_trial,
                            "total_trials": search.iterations,
                            "progress": {},
                            "heartbeat": True
                        }
                    )
                    logger.info(f"Heartbeat WebSocket enviado para a busca {search_id}")
                
                # Se a task terminou, verificar resultado
                if task.ready():
                    if task.successful():
                        result = task.get()
                        logger.info(f"Busca {search_id} concluída com sucesso: {result}")
                        # Atualizar best_metrics com as novas métricas
                        if result.get("best_metrics"):
                            search.best_metrics = {
                                **search.best_metrics,  # Manter métricas existentes
                                "precision": result["best_metrics"].get("precision"),
                                "recall": result["best_metrics"].get("recall"),
                                "f1_score": result["best_metrics"].get("f1_score"),
                                "best_precision": result["best_metrics"].get("best_precision"),
                                "best_recall": result["best_metrics"].get("best_recall"),
                                "best_f1_score": result["best_metrics"].get("best_f1_score")
                            }
                            self._db.commit()
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
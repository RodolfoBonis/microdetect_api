import os
import json
import logging
import asyncio
import random
import time
from celery import Task, shared_task, current_task
from microdetect.core.celery_app import celery_app
from microdetect.services.yolo_service import YOLOService
from microdetect.models.hyperparam_search import HyperparamSearch
from microdetect.database.database import SessionLocal, get_db
from microdetect.core.hyperparam_core import (
    prepare_hyperparam_directory,
    prepare_hyperparam_config,
    RandomSearchOptimizer
)
from datetime import datetime
from microdetect.core.config import settings
from microdetect.services.dataset_service import DatasetService
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class HyperparamTask(Task):
    _yolo_service = None
    
    @property
    def yolo_service(self):
        if self._yolo_service is None:
            self._yolo_service = YOLOService()
        return self._yolo_service

@shared_task(bind=True)
def run_hyperparameter_search(
    self,
    search_id: int,
    dataset_id: int,
    param_space: Dict[str, Any],
    model_type: str,
    model_version: str,
    n_trials: int,
    search_algorithm: str,
    objective_metric: str = "map",
    data_yaml_path: str = None
) -> Dict[str, Any]:
    """
    Tarefa Celery para executar a busca de hiperparâmetros.
    
    Args:
        self: A tarefa atual
        search_id: ID da busca de hiperparâmetros
        dataset_id: ID do dataset
        param_space: Espaço de parâmetros para otimização
        model_type: Tipo do modelo (ex: "yolov8")
        model_version: Versão do modelo (ex: "n", "s", "m", "l", "x")
        n_trials: Número de tentativas
        search_algorithm: Algoritmo de busca ("random", "tpe")
        objective_metric: Métrica a ser otimizada ("map", "map50", etc.)
        data_yaml_path: Caminho para o arquivo data.yaml (opcional)
        
    Returns:
        Melhores hiperparâmetros e métricas
    """
    try:
        logger.info(f"Iniciando busca de hiperparâmetros ID: {search_id}, Dataset: {dataset_id}")
        
        # Criar serviços necessários
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        db = SessionLocal()
        try:
            # Verificar se a busca existe
            search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
            if not search:
                raise ValueError(f"Busca de hiperparâmetros ID {search_id} não encontrada")
            
            # Atualizar o status como executando
            search.status = "running"
            db.commit()
            
            dataset_service = DatasetService(db)
            yolo_service = YOLOService()
            
            # Preparar dataset para treinamento, se necessário
            if not data_yaml_path:
                data_yaml_path = dataset_service.prepare_for_training(dataset_id)
            
            # Lista para armazenar os resultados das tentativas
            trials_results = []
            
            # Função para atualizar o estado da tarefa
            def update_state(metrics, trial_num=None, total_trials=n_trials, progress_type="trial"):
                """Atualiza o estado da tarefa Celery."""
                if trial_num is not None:
                    state_info = {
                        "current_trial": trial_num,
                        "total_trials": total_trials,
                        "progress_type": progress_type,
                        **metrics
                    }
                else:
                    state_info = {
                        "progress_type": progress_type,
                        **metrics
                    }
                
                # Atualizar o estado da tarefa Celery
                self.update_state(
                    state="PROGRESS",
                    meta=state_info
                )
                
                # Para depuração, registrar o estado sendo enviado
                logger.debug(f"Estado atualizado: {state_info}")
            
            # Função para monitorar e reportar o progresso durante o treinamento
            def create_training_progress_callback(trial_num):
                """Cria um callback para monitorar o progresso do treinamento."""
                def training_progress_callback(metrics):
                    """Reporta progresso de treinamento durante cada trial."""
                    try:
                        # Verificar se metrics é um dicionário
                        if not isinstance(metrics, dict):
                            logger.error(f"Metrics não é um dicionário: {type(metrics)}")
                            return
                            
                        # Adicionar informações do trial ao progresso
                        metrics["progress_type"] = "epoch_in_trial"
                        metrics["current_trial"] = trial_num
                        metrics["total_trials"] = n_trials
                        
                        # Registrar para depuração
                        logger.info(f"Callback de treinamento chamado: trial={trial_num}, tipo={metrics.get('progress_type')}, época={metrics.get('epoch')}")
                        
                        # Atualizar o estado
                        self.update_state(
                            state="PROGRESS",
                            meta=metrics
                        )
                        
                        # Para depuração, registrar o progresso
                        logger.debug(f"Progresso de treinamento: {metrics}")
                    except Exception as e:
                        logger.error(f"Erro no callback de progresso: {str(e)}")
                        logger.error(f"Tipo do erro: {type(e)}")
                        logger.error(f"Traceback completo: {e.__traceback__}")
                
                return training_progress_callback
            
            # Implementação simples de busca aleatória
            best_params = None
            best_metric_value = float('inf') if objective_metric == "loss" else float('-inf')
            
            # Definir uma seed para reprodutibilidade
            random.seed(42)
            
            # Para cada tentativa
            for current_trial in range(n_trials):
                # Gerar parâmetros aleatórios
                trial_params = {}
                for param_name, param_config in param_space.items():
                    logger.debug(f"Processando parâmetro: {param_name}, config: {param_config}")
                    
                    # Processamento para intervalo min/max
                    if isinstance(param_config, dict) and "min" in param_config and "max" in param_config:
                        # Mapear para parâmetros específicos do YOLOv8
                        if param_name == "learning_rate":
                            # Para learning_rate, usar lr0
                            trial_params["lr0"] = random.uniform(param_config["min"], param_config["max"])
                            logger.info(f"Definindo lr0={trial_params['lr0']} a partir de {param_name}")
                        elif param_name == "batch_size":
                            # Para batch_size, usar batch e garantir inteiro
                            trial_params["batch"] = int(random.uniform(param_config["min"], param_config["max"]))
                            logger.info(f"Definindo batch={trial_params['batch']} a partir de {param_name}")
                        elif param_name == "epochs":
                            # Para epochs, usar diretamente
                            trial_params["epochs"] = int(random.uniform(param_config["min"], param_config["max"]))
                            logger.info(f"Definindo epochs={trial_params['epochs']} a partir de {param_name}")
                        else:
                            # Para outros parâmetros, escolher um valor apropriado
                            if isinstance(param_config["min"], int) and isinstance(param_config["max"], int):
                                trial_params[param_name] = random.randint(param_config["min"], param_config["max"])
                            else:
                                trial_params[param_name] = random.uniform(param_config["min"], param_config["max"])
                            logger.info(f"Definindo {param_name}={trial_params[param_name]} a partir de intervalo")
                    
                    # Processamento para parâmetros com "type"
                    elif isinstance(param_config, dict) and "type" in param_config:
                        if param_config["type"] == "categorical":
                            trial_params[param_name] = random.choice(param_config["values"])
                            logger.info(f"Definindo {param_name}={trial_params[param_name]} (categórico)")
                        elif param_name == "int":
                            trial_params[param_name] = random.randint(param_config["min"], param_config["max"])
                            logger.info(f"Definindo {param_name}={trial_params[param_name]} (inteiro)")
                        elif param_config["type"] == "float":
                            trial_params[param_name] = random.uniform(param_config["min"], param_config["max"])
                            logger.info(f"Definindo {param_name}={trial_params[param_name]} (float)")
                    
                    # Processamento para lista de valores (assumir categórico)
                    elif isinstance(param_config, list):
                        trial_params[param_name] = random.choice(param_config)
                        logger.info(f"Definindo {param_name}={trial_params[param_name]} (lista de valores)")
                    
                    # Valor direto
                    elif isinstance(param_config, (int, float, str, bool)):
                        trial_params[param_name] = param_config
                        logger.info(f"Definindo {param_name}={trial_params[param_name]} (valor direto)")
                    
                    # Outros casos (não processáveis)
                    else:
                        logger.warning(f"Formato não reconhecido para parâmetro {param_name}: {param_config}")
                        # Não incluir parâmetros que não podem ser processados
                
                # Verificar se CUDA está disponível e adicionar device
                if not settings.USE_CUDA:
                    logger.info("CUDA não disponível. Adicionando device=cpu para treinamento.")
                    trial_params["device"] = "cpu"
                else:
                    logger.info("CUDA disponível. Usando GPU para treinamento.")
                    trial_params["device"] = "0"  # ou outro índice de GPU apropriado
                
                # Atualizar o estado com os parâmetros sugeridos
                update_state(
                    {
                        "trial": current_trial + 1,
                        "total_trials": n_trials,
                        "params": trial_params
                    }
                )
                
                try:
                    # Realizar treinamento com os parâmetros desta tentativa
                    metrics = loop.run_until_complete(
                        yolo_service.train(
                            dataset_id=dataset_id,
                            model_type=model_type,
                            model_version=model_version,
                            hyperparameters=trial_params,
                            callback=create_training_progress_callback(current_trial + 1),  # Passar callback para monitorar progresso
                            db_session=db,
                            data_yaml_path=data_yaml_path
                        )
                    )
                    
                    # Registrar os resultados
                    trial_result = {
                        "params": trial_params,
                        "metrics": metrics
                    }
                    trials_results.append(trial_result)
                    
                    search.trials_data = trials_results
                    
                    # Verificar se este é o melhor resultado
                    metric_value = metrics.get(objective_metric, 0)
                    
                    if objective_metric == "loss":
                        # Para loss, menor é melhor
                        if metric_value < best_metric_value:
                            best_metric_value = metric_value
                            best_params = trial_params
                    else:
                        # Para outras métricas, maior é melhor
                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_params = trial_params
                    
                    # Atualizar os melhores parâmetros e métricas no banco
                    search.best_params = best_params
                    search.best_metrics = metrics
                    
                    db.commit()
                    
                    # Atualizar o estado com os resultados
                    update_state(
                        {
                            "trial": current_trial + 1,
                            "total_trials": n_trials,
                            "params": trial_params,
                            "metrics": metrics,
                            "best_params": best_params,
                            "best_metrics": search.best_metrics
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Erro durante o trial {current_trial + 1}: {str(e)}")
                    
                    # Atualizar o estado com erro
                    update_state(
                        {
                            "status": f"Erro no trial {current_trial + 1}/{n_trials}: {str(e)}",
                            "error": str(e)
                        },
                        trial_num=current_trial + 1,
                        progress_type="trial_error"
                    )
                    
                    # Continuar para a próxima tentativa
                    continue
            
            # Criar resultado final
            result = {
                "best_params": best_params,
                "best_metric": best_metric_value,
                "objective_metric": objective_metric,
                "trials": trials_results
            }
            
            # Atualizar o status da busca
            search.status = "completed"
            search.completed_at = datetime.utcnow()
            search.result = json.dumps(result)
            db.commit()
            
            logger.info(f"Busca de hiperparâmetros concluída. ID: {search_id}")
            return result
            
        except Exception as e:
            logger.error(f"Erro durante otimização: {str(e)}")
            
            # Atualizar o status de erro na busca
            if 'search' in locals():
                search.status = "failed"
                search.error_message = str(e)
                db.commit()
            
            # Repassar a exceção
            raise
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Erro durante otimização: {str(e)}")
        raise

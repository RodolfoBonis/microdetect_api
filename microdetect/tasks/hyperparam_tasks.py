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
    param_space: List[Dict[str, Any]],
    model_type: str,
    model_version: str,
    data_yaml_path: str = None
) -> Dict[str, Any]:
    """
    Tarefa Celery para executar a busca de hiperparâmetros.
    
    Args:
        self: A tarefa atual
        search_id: ID da busca de hiperparâmetros
        dataset_id: ID do dataset
        param_space: Lista de combinações de parâmetros para testar
        model_type: Tipo do modelo (ex: "yolov8")
        model_version: Versão do modelo (ex: "n", "s", "m", "l", "x")
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
            def update_state(metrics, trial_num=None, total_trials=len(param_space), progress_type="trial"):
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
                        metrics["total_trials"] = len(param_space)
                        
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
            
            # Implementação de busca em grade (grid search)
            best_params = None
            best_metric_value = float('-inf')  # Usando mAP como métrica padrão
            
            # Para cada combinação de parâmetros
            for current_trial, trial_params in enumerate(param_space):
                try:
                    # Atualizar o estado com os parâmetros sugeridos
                    update_state(
                        {
                            "trial": current_trial + 1,
                            "total_trials": len(param_space),
                            "params": trial_params
                        }
                    )
                    
                    # Realizar treinamento com os parâmetros desta tentativa
                    metrics = loop.run_until_complete(
                        yolo_service.train(
                            dataset_id=dataset_id,
                            model_type=trial_params.get("model_type", model_type),
                            model_version=trial_params.get("model_size", model_version),
                            hyperparameters=trial_params,
                            callback=create_training_progress_callback(current_trial + 1),
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
                    metric_value = metrics.get("map50", 0)  # Usando mAP50 como métrica principal
                    
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
                            "total_trials": len(param_space),
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
                            "status": f"Erro no trial {current_trial + 1}/{len(param_space)}: {str(e)}",
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

import os
import shutil
import json
import logging
import random
import asyncio
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

# Configurar logging
logger = logging.getLogger(__name__)

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
        Inicia a busca de hiperparâmetros.
        
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
        
        # Criar diretório para treinamento
        training_dir = settings.TRAINING_DIR / f"hyperparam_search_{search_id}"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar se foi especificado um dispositivo para a busca
        device = search.search_space.get("device", "auto")
        
        # Se device foi especificado no search_space, remover para não interferir na busca
        if "device" in search.search_space:
            logger.info(f"Using device '{device}' for all iterations")
            # Fazemos uma cópia para não modificar o objeto original no banco
            search_space = search.search_space.copy()
            search_space.pop("device", None)
        else:
            search_space = search.search_space
        
        # Iniciar otimizador
        optimizer = HyperparameterOptimizer(
            search_space=search_space,
            iterations=search.iterations
        )
        
        try:
            # Executar iterações de busca
            for i in range(search.iterations):
                logger.info(f"Starting iteration {i+1}/{search.iterations} for search {search_id}")
                
                # Gerar parâmetros para esta iteração
                params = optimizer.generate_params()
                
                # Adicionar o dispositivo a ser usado
                params["device"] = device
                
                logger.info(f"Generated params: {params}")
                
                # Criar diretório para esta iteração
                iteration_dir = training_dir / f"iteration_{i+1}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                
                # Treinar modelo com estes parâmetros
                metrics = await self._train_iteration(
                    params=params,
                    dataset_id=search.dataset_id,
                    output_dir=iteration_dir
                )
                
                # Atualizar melhor resultado
                optimizer.update_best(params, metrics)
                
                # Atualizar dados no banco
                search.trials_data = optimizer.get_trials_data()
                search.best_params = optimizer.get_best_params()
                search.best_metrics = optimizer.get_best_metrics()
                db.commit()
                
                # Limpar diretório desta iteração para economizar espaço
                # (exceto o melhor modelo, que é mantido separadamente)
                if i < search.iterations - 1:  # Manter a última iteração para debug
                    shutil.rmtree(iteration_dir, ignore_errors=True)
            
            # Criar modelo final com os melhores parâmetros
            logger.info(f"Training final model with best params for search {search_id}")
            
            # Garantir que o dispositivo seja preservado nos melhores parâmetros
            best_params = optimizer.get_best_params()
            if "device" not in best_params:
                best_params["device"] = device
                
            final_training_session = await self._create_final_model(
                search_id=search_id,
                dataset_id=search.dataset_id,
                best_params=best_params,
                db=db
            )
            
            # Atualizar com a referência à sessão de treinamento final
            search.training_session_id = final_training_session.id
            search.status = HyperparamSearchStatus.COMPLETED
            search.completed_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error in hyperparameter search {search_id}: {e}")
            search.status = HyperparamSearchStatus.FAILED
            search.completed_at = datetime.utcnow()
        finally:
            # Salvar alterações finais
            db.commit()
            
            # Limpar diretório de busca
            shutil.rmtree(training_dir, ignore_errors=True)
    
    async def _train_iteration(
        self,
        params: Dict[str, Any],
        dataset_id: int,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Treina um modelo para uma iteração da busca.
        
        Args:
            params: Hiperparâmetros a utilizar
            dataset_id: ID do dataset
            output_dir: Diretório de saída
            
        Returns:
            Métricas de treinamento
        """
        # Extrair model_type e model_size antes de configurar os parâmetros
        model_type = params.pop("model_type", "yolov8")
        model_size = params.pop("model_size", "n")
        
        # Configurar parâmetros de treinamento com nomes corretos para YOLO
        train_params = {
            "epochs": params.get("epochs", 10),  # Reduzir épocas para busca mais rápida
            "batch": params.get("batch_size", 16),  # YOLO usa 'batch', não 'batch_size'
            "imgsz": params.get("imgsz", 640),
            "device": params.get("device", "auto"),
            "project": str(output_dir.parent),
            "name": output_dir.name,
            "exist_ok": True
        }
        
        # Remover batch_size se estiver presente para evitar duplicação
        if "batch_size" in params:
            params.pop("batch_size")
        
        # Incluir outros parâmetros específicos
        for key, value in params.items():
            if key not in train_params:
                train_params[key] = value
        
        # Treinar modelo
        metrics = await self.yolo_service.train(
            dataset_id=dataset_id,
            model_type=model_type,
            model_version=model_size,
            hyperparameters=train_params,
            callback=None  # Não precisamos de callback para iterações de busca
        )
        
        return metrics
    
    async def _create_final_model(
        self,
        search_id: int,
        dataset_id: int,
        best_params: Dict[str, Any],
        db: Session
    ) -> TrainingSession:
        """
        Cria o modelo final usando os melhores hiperparâmetros encontrados.
        
        Args:
            search_id: ID da busca
            dataset_id: ID do dataset
            best_params: Melhores hiperparâmetros
            db: Sessão do banco de dados
            
        Returns:
            Sessão de treinamento criada
        """
        # Extrair informações do modelo
        model_type = best_params.get("model_type", "yolov8")
        model_size = best_params.get("model_size", "n")
        
        # Criar sessão de treinamento
        training_session = TrainingSession(
            name=f"Modelo final da busca {search_id}",
            description=f"Modelo treinado com os melhores hiperparâmetros da busca {search_id}",
            model_type=model_type,
            model_version=model_size,
            hyperparameters=best_params,
            dataset_id=dataset_id,
            status=TrainingStatus.PENDING
        )
        
        db.add(training_session)
        db.commit()
        db.refresh(training_session)
        
        # Atualizar status
        training_session.status = TrainingStatus.RUNNING
        training_session.started_at = datetime.utcnow()
        db.commit()
        
        try:
            # Treinar modelo final (épocas completas)
            params = best_params.copy()
            params["epochs"] = params.get("epochs", 100)  # Épocas completas para o modelo final
            
            # Configurar diretório de saída
            model_dir = settings.TRAINING_DIR / f"model_{training_session.id}"
            params["project"] = str(model_dir.parent)
            params["name"] = model_dir.name
            params["exist_ok"] = True
            
            # Corrigir nomes de parâmetros para YOLO
            if "batch_size" in params:
                params["batch"] = params.pop("batch_size")
                
            # Remover parâmetros que não são válidos para o YOLO
            if "model_type" in params:
                params.pop("model_type")
                
            if "model_size" in params:
                params.pop("model_size")
            
            # Treinar modelo
            metrics = await self.yolo_service.train(
                dataset_id=dataset_id,
                model_type=model_type,
                model_version=model_size,
                hyperparameters=params,
                callback=None  # Não precisamos de callback para o modelo final
            )
            
            # Atualizar métricas e status
            training_session.metrics = metrics
            training_session.status = TrainingStatus.COMPLETED
            training_session.completed_at = datetime.utcnow()
            
            # Copiar modelo para pasta de modelos
            model_path = model_dir / "weights" / "best.pt"
            if model_path.exists():
                model_name = f"{model_type}{model_size}"
                target_path = settings.MODELS_DIR / f"{model_name}_{training_session.id}.pt"
                shutil.copy(model_path, target_path)
                
        except Exception as e:
            logger.error(f"Error in final model training for search {search_id}: {e}")
            training_session.status = TrainingStatus.FAILED
            training_session.completed_at = datetime.utcnow()
        finally:
            # Salvar alterações
            db.commit()
            
        return training_session
    
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
import os
import shutil
import json
import logging
import random
import asyncio
import concurrent.futures
import subprocess
import sys
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
        self._search_tasks = {}
        self._db = next(get_db())  # Obter uma sessão do banco para usar nos métodos
        self._progress_data = {}  # Armazenar dados de progresso em tempo real
    
    def __del__(self):
        """Fecha a sessão do banco quando o serviço for destruído."""
        try:
            self._db.close()
        except:
            pass
    
    def get_progress(self, search_id: int) -> Dict[str, Any]:
        """
        Obtém os dados de progresso em tempo real de uma busca.
        
        Args:
            search_id: ID da busca
            
        Returns:
            Dados de progresso da busca
        """
        if search_id not in self._progress_data:
            return {
                "current_iteration": 0,
                "total_iterations": 0,
                "iterations_completed": 0,
                "current_params": {},
                "best_params": {},
                "best_metrics": {},
                "trials": [],
                "status": "pending"
            }
        return self._progress_data[search_id]
    
    def update_progress(self, search_id: int, progress_data: Dict[str, Any]):
        """
        Atualiza os dados de progresso em tempo real de uma busca.
        
        Args:
            search_id: ID da busca
            progress_data: Novos dados de progresso
        """
        if search_id not in self._progress_data:
            self._progress_data[search_id] = {
                "current_iteration": 0,
                "total_iterations": 0,
                "iterations_completed": 0,
                "current_params": {},
                "best_params": {},
                "best_metrics": {},
                "trials": [],
                "status": "pending"
            }
        
        # Atualizar com novos dados
        self._progress_data[search_id].update(progress_data)
        
        # Atualizar no banco de dados se tiver dados significativos
        try:
            if progress_data.get("best_params") and progress_data.get("best_metrics"):
                search = self._db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
                if search:
                    search.best_params = progress_data["best_params"]
                    search.best_metrics = progress_data["best_metrics"]
                    # Verificar se temos dados de trials
                    if "trials" in progress_data:
                        search.trials_data = progress_data["trials"]
                    self._db.commit()
        except Exception as e:
            logger.error(f"Erro ao atualizar progresso no banco: {e}")
    
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
        Inicia a busca de hiperparâmetros em um processo separado.
        
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
        
        # Inicializar dados de progresso
        self._progress_data[search_id] = {
            "current_iteration": 0,
            "total_iterations": search.iterations,
            "iterations_completed": 0,
            "status": "running"
        }
        
        # Criar diretório para treinamento
        training_dir = settings.TRAINING_DIR / f"hyperparam_search_{search_id}"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar se foi especificado um dispositivo para a busca
        device = search.search_space.get("device", "auto")
        
        # Forçar CPU se CUDA não estiver disponível e device for auto
        if device == "auto" and not CUDA_AVAILABLE:
            device = "cpu"
            logger.info("CUDA não disponível. Forçando device=cpu para a busca.")
        
        # Se device foi especificado no search_space, remover para não interferir na busca
        if "device" in search.search_space:
            logger.info(f"Using device '{device}' for all iterations")
            # Fazemos uma cópia para não modificar o objeto original no banco
            search_space = search.search_space.copy()
            search_space.pop("device", None)
        else:
            search_space = search.search_space
        
        # Criar arquivo para comunicação entre processos
        progress_file = training_dir / "progress.json"
        with open(progress_file, "w") as f:
            json.dump(self._progress_data[search_id], f)
        
        # Salvar configuração para o processo em um arquivo
        config_path = training_dir / "search_config.json"
        config = {
            "search_id": search_id,
            "dataset_id": search.dataset_id,
            "search_space": search_space,
            "iterations": search.iterations,
            "device": device,
            "output_dir": str(training_dir),
            "progress_file": str(progress_file)
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Iniciar o processo em background
        try:
            # Criar tarefa para executar a busca em um processo separado usando subprocess
            # Isso evita problemas de pickle com objetos de thread
            cmd = [
                sys.executable,
                "-c",
                f"""
import json
import asyncio
import sys
import torch
import time
from pathlib import Path

# Carregar configuração
config_path = "{config_path}"
with open(config_path, "r") as f:
    config = json.load(f)

# Configurar caminho no Python
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

# Importar serviços
from microdetect.services.hyperparam_service import HyperparameterOptimizer
from microdetect.services.yolo_service import YOLOService
from microdetect.core.config import settings
from microdetect.services.dataset_service import DatasetService
from microdetect.models.dataset import Dataset
from microdetect.database.database import get_db
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config["output_dir"]) / "search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hyperparam_search")

# Verificar CUDA
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available in subprocess: {{cuda_available}}")

# Verificar e ajustar o dispositivo
device = config.get("device", "auto")
if device == "auto" and not cuda_available:
    device = "cpu"
    logger.info("CUDA não disponível no processo. Forçando device=cpu.")
    config["device"] = "cpu"

# Função para atualizar o arquivo de progresso
def update_progress_file(progress_data):
    try:
        # Ler dados atuais
        with open(config["progress_file"], "r") as f:
            current_data = json.load(f)
        
        # Atualizar com novos dados
        current_data.update(progress_data)
        
        # Escrever de volta
        with open(config["progress_file"], "w") as f:
            json.dump(current_data, f)
    except Exception as e:
        logger.error(f"Erro ao atualizar arquivo de progresso: {{e}}")

async def run_search():
    try:
        # Criar diretório de saída
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar serviços
        yolo_service = YOLOService()
        
        # Obter sessão do banco de dados
        db = next(get_db())
        
        # Obter dataset e preparar para treinamento
        dataset_service = DatasetService(db)
        try:
            dataset = db.query(Dataset).filter(Dataset.id == config["dataset_id"]).first()
            if not dataset:
                raise ValueError(f"Dataset {{config['dataset_id']}} não encontrado")
            
            # Preparar dataset e obter o caminho correto para data.yaml
            data_yaml_path = dataset_service.prepare_for_training(config["dataset_id"])
            logger.info(f"Dataset preparado. Usando arquivo data.yaml: {{data_yaml_path}}")
        except Exception as e:
            logger.error(f"Erro ao preparar dataset: {{e}}")
            raise
        
        # Inicializar otimizador
        optimizer = HyperparameterOptimizer(
            search_space=config["search_space"],
            iterations=config["iterations"]
        )
        
        # Atualizar status inicial
        update_progress_file({{
            "total_iterations": config["iterations"],
            "current_iteration": 0,
            "iterations_completed": 0,
            "status": "running"
        }})
        
        # Executar iterações
        for i in range(config["iterations"]):
            print(f"Starting iteration {{i+1}}/{{config['iterations']}}")
            
            # Gerar parâmetros
            params = optimizer.generate_params()
            params["device"] = config["device"]  # Usar o dispositivo ajustado
            
            # Atualizar progresso
            update_progress_file({{
                "current_iteration": i+1,
                "current_params": params
            }})
            
            # Criar diretório para a iteração
            iteration_dir = output_dir / f"iteration_{{i+1}}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Treinar modelo
                model_type = params.get("model_type", "yolov8")
                model_version = params.get("model_version", "n")
                
                # Limpar parâmetros
                clean_params = {{}}
                for key, value in params.items():
                    if key not in ["model_type", "model_version", "model_size"]:
                        if key == "epochs" and value is not None:
                            try:
                                clean_params[key] = int(value)
                            except (ValueError, TypeError):
                                clean_params[key] = 10
                        else:
                            clean_params[key] = value
                
                # Configurar diretório de saída
                clean_params["project"] = str(iteration_dir.parent)
                clean_params["name"] = iteration_dir.name
                clean_params["exist_ok"] = True
                
                # Verificar device novamente
                if clean_params.get("device") == "auto" and not cuda_available:
                    clean_params["device"] = "cpu"
                
                # Treinar modelo
                logger.info(f"Training with params: {{clean_params}}")
                metrics = await yolo_service.train(
                    dataset_id=config["dataset_id"],
                    model_type=model_type,
                    model_version=model_version,
                    hyperparameters=clean_params,
                    data_yaml_path=data_yaml_path  # Passar o caminho do data.yaml
                )
                
                # Atualizar resultado
                optimizer.update_best(params, metrics)
                
                # Atualizar progresso
                trial = {{"params": params, "metrics": metrics, "iteration": i+1}}
                update_progress_file({{
                    "iterations_completed": i+1,
                    "best_params": optimizer.get_best_params(),
                    "best_metrics": optimizer.get_best_metrics(),
                    "trials": optimizer.get_trials_data(),
                    "current_params": params,
                    "current_metrics": metrics
                }})
                
                # Salvar resultados
                with open(iteration_dir / "results.json", "w") as f:
                    json.dump(trial, f)
                    
            except Exception as e:
                print(f"Error in iteration {{i+1}}: {{str(e)}}")
                # Atualizar progresso mesmo com erro
                update_progress_file({{
                    "error": str(e)
                }})
        
        # Salvar resultados finais
        results = {{
            "best_params": optimizer.get_best_params(),
            "best_metrics": optimizer.get_best_metrics(),
            "trials": optimizer.get_trials_data()
        }}
        
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(results, f)
        
        # Atualizar status final
        update_progress_file({{
            "status": "completed",
            "best_params": optimizer.get_best_params(),
            "best_metrics": optimizer.get_best_metrics(),
            "trials": optimizer.get_trials_data(),
            "iterations_completed": config["iterations"]
        }})
            
        print("Search completed successfully")
            
    except Exception as e:
        print(f"Error in search: {{str(e)}}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        # Atualizar status em caso de erro
        update_progress_file({{
            "status": "failed",
            "error": str(e)
        }})

# Executar a busca
asyncio.run(run_search())
                """
            ]
            
            # Iniciar o processo
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Configurar monitoramento assíncrono do processo
            self._search_tasks[search_id] = {
                "process": process,
                "status": "running",
                "output_dir": training_dir,
                "progress_file": progress_file
            }
            
            # Iniciar tarefa para monitorar a conclusão
            asyncio.create_task(self._monitor_search_completion(search_id, db))
            
            # Iniciar tarefa para monitorar o arquivo de progresso
            asyncio.create_task(self._monitor_progress_file(search_id))
            
        except Exception as e:
            logger.error(f"Error starting search process: {str(e)}")
            search.status = HyperparamSearchStatus.FAILED
            search.error_message = str(e)
            search.completed_at = datetime.utcnow()
            db.commit()
    
    async def _monitor_progress_file(self, search_id: int):
        """
        Monitora o arquivo de progresso para atualizar os dados em tempo real.
        """
        if search_id not in self._search_tasks:
            return
        
        progress_file = self._search_tasks[search_id]["progress_file"]
        if not os.path.exists(progress_file):
            logger.error(f"Arquivo de progresso não encontrado: {progress_file}")
            return
        
        try:
            # Verificar o arquivo periodicamente
            while search_id in self._search_tasks and self._search_tasks[search_id]["status"] in ["running", "starting"]:
                try:
                    # Ler arquivo de progresso
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                    
                    # Atualizar dados de progresso
                    self.update_progress(search_id, progress_data)
                    
                    # Verificar se a busca terminou
                    if progress_data.get("status") in ["completed", "failed"]:
                        break
                        
                except json.JSONDecodeError:
                    # Arquivo pode estar sendo escrito, ignorar erro
                    pass
                except Exception as e:
                    logger.error(f"Erro ao ler arquivo de progresso: {e}")
                
                # Aguardar próxima verificação
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Erro ao monitorar arquivo de progresso: {e}")
    
    async def _monitor_search_completion(self, search_id: int, db: Session):
        """
        Monitora a conclusão de uma tarefa de busca e atualiza o banco de dados.
        """
        if search_id not in self._search_tasks:
            return
            
        try:
            # Obter processo
            process = self._search_tasks[search_id]["process"]
            output_dir = Path(self._search_tasks[search_id]["output_dir"])
            
            # Esperar pelo processo
            stdout, stderr = await asyncio.get_event_loop().run_in_executor(
                None, process.communicate
            )
            
            # Registrar saída
            logger.info(f"Search process for {search_id} completed with code {process.returncode}")
            logger.debug(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
            
            if process.returncode != 0:
                logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                raise RuntimeError(f"Search process failed with code {process.returncode}")
            
            # Verificar se o arquivo de resultado existe
            results_file = output_dir / "final_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found at {results_file}")
            
            # Carregar resultados
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Obter busca do banco
            search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
            if not search:
                return
                
            # Atualizar resultados e status
            search.best_params = results.get("best_params", {})
            search.best_metrics = results.get("best_metrics", {})
            search.trials_data = results.get("trials", [])
            search.status = HyperparamSearchStatus.COMPLETED
            search.completed_at = datetime.utcnow()
            db.commit()
            
            # Atualizar dados de progresso
            self.update_progress(search_id, {
                "status": "completed",
                "best_params": search.best_params,
                "best_metrics": search.best_metrics,
                "trials": search.trials_data,
                "iterations_completed": search.iterations
            })
            
            # Criar modelo final com os melhores parâmetros
            await self._create_final_model(
                search_id=search_id,
                dataset_id=search.dataset_id,
                best_params=search.best_params,
                db=db
            )
            
            # Limpar tarefa
            self._search_tasks[search_id]["status"] = "completed"
            
        except Exception as e:
            # Obter busca do banco
            search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
            if search:
                # Atualizar status em caso de erro
                search.status = HyperparamSearchStatus.FAILED
                search.error_message = str(e)
                search.completed_at = datetime.utcnow()
                db.commit()
                
            # Atualizar dados de progresso
            self.update_progress(search_id, {
                "status": "failed", 
                "error": str(e)
            })
                
            # Limpar tarefa
            self._search_tasks[search_id]["status"] = "failed"
            logger.error(f"Error monitoring search completion: {str(e)}")
            
        finally:
            # Remover tarefa após um tempo
            await asyncio.sleep(60)  # Manter status por um minuto
            if search_id in self._search_tasks:
                del self._search_tasks[search_id]
            # Remover dados de progresso
            await asyncio.sleep(300)  # Manter status por mais tempo
            if search_id in self._progress_data:
                del self._progress_data[search_id]
    
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
        try:
            # Garantir que best_params é um dicionário válido
            if not best_params or not isinstance(best_params, dict):
                logger.warning(f"Best params inválidos para a busca {search_id}: {best_params}")
                best_params = {}
                
            # Extrair informações do modelo com valores padrão seguros
            model_type = best_params.get("model_type", "yolov8")
            model_size = best_params.get("model_version", "n")
            
            # Preparar parâmetros para o treinamento final
            # Fazer uma cópia profunda dos parâmetros para evitar modificar o original
            params = {}
            for key, value in best_params.items():
                params[key] = value
            
            # Verificar e ajustar o dispositivo se necessário
            if "device" in params and params["device"] == "auto" and not CUDA_AVAILABLE:
                params["device"] = "cpu"
                logger.info("CUDA não disponível. Forçando device=cpu para o modelo final.")
            elif "device" not in params and not CUDA_AVAILABLE:
                params["device"] = "cpu"
                logger.info("CUDA não disponível. Adicionando device=cpu para o modelo final.")
            
            # Definir parâmetros explicitamente com valores padrão se não existirem
            if "epochs" not in params:
                params["epochs"] = 100
            else:
                # Garantir que epochs seja um inteiro
                try:
                    params["epochs"] = int(params["epochs"])
                except (ValueError, TypeError):
                    logger.warning(f"Valor de epochs inválido: {params.get('epochs')}. Usando valor padrão 100.")
                    params["epochs"] = 100
            
            # Criar instância do TrainingService
            from microdetect.services.training_service import TrainingService
            training_service = TrainingService()
            
            # Criar sessão de treinamento
            training_session = await training_service.create_training_session(
                dataset_id=dataset_id,
                model_type=model_type,
                model_version=model_size,
                name=f"Modelo final da busca {search_id}",
                description=f"Modelo treinado com os melhores hiperparâmetros da busca {search_id}",
                hyperparameters=params
            )
            
            # Iniciar treinamento em background
            await training_service.start_training(training_session.id)
            
            # Atualizar a busca com o ID da sessão de treinamento final
            search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
            if search:
                search.training_session_id = training_session.id
                db.commit()
            
            logger.info(f"Treinamento do modelo final iniciado em background para a busca {search_id}")
            
            return training_session
            
        except Exception as e:
            logger.error(f"Error in final model training for search {search_id}: {e}")
            if 'training_session' in locals():
                training_session.status = TrainingStatus.FAILED
                training_session.error_message = str(e)
                training_session.completed_at = datetime.utcnow()
            else:
                # Criar sessão com status de falha se não foi possível criar antes
                training_session = TrainingSession(
                    name=f"Modelo final da busca {search_id} (falhou)",
                    description=f"Tentativa falhou: {str(e)}",
                    model_type="yolov8",
                    model_version="n",
                    hyperparameters={},
                    dataset_id=dataset_id,
                    status=TrainingStatus.FAILED,
                )
                db.add(training_session)
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
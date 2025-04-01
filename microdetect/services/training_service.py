import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.models.model import Model
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService
from sqlalchemy.orm import Session
from microdetect.database.database import get_db
import shutil
import asyncio
import subprocess
import sys
import logging
import torch
from microdetect.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

# Verificar se CUDA está disponível
CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(f"CUDA available: {CUDA_AVAILABLE}")

class TrainingService:
    def __init__(self):
        self.training_dir = settings.TRAINING_DIR
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_service = YOLOService()
        self._training_tasks = {}
        self._db = next(get_db())  # Obter uma sessão do banco para usar nos métodos
        self._progress_data = {}  # Armazenar dados de progresso em tempo real

    async def create_training_session(
        self,
        dataset_id: int,
        model_type: str,
        model_version: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainingSession:
        """
        Cria uma nova sessão de treinamento.
        
        Args:
            dataset_id: ID do dataset
            model_type: Tipo do modelo (ex: "yolov8")
            model_version: Versão do modelo
            name: Nome da sessão (opcional)
            description: Descrição da sessão (opcional)
            hyperparameters: Parâmetros de treinamento (opcional)
            
        Returns:
            Objeto TrainingSession criado
        """
        # Verificar dataset
        dataset = self._db.query(Dataset).get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Criar diretório da sessão
        session_dir = self.training_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True)
        
        # Criar registro no banco
        session = TrainingSession(
            name=name or f"Treinamento {dataset.name}",
            description=description,
            dataset_id=dataset_id,
            model_type=model_type,
            model_version=model_version,
            hyperparameters=hyperparameters or {},
            status="pending",
            metrics={},
            log_file=str(session_dir / "training.log")
        )
        
        # Adicionar e salvar no banco
        self._db.add(session)
        self._db.commit()
        self._db.refresh(session)
        
        # Salvar configuração
        config = {
            "dataset": {
                "id": dataset_id,
                "name": dataset.name,
                "classes": dataset.classes
            },
            "model": {
                "type": model_type,
                "version": model_version
            },
            "hyperparameters": session.hyperparameters,
            "created_at": session.created_at.isoformat()
        }
        
        with open(session_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        return session

    async def get_training_session(self, session_id: int) -> TrainingSession:
        """
        Recupera uma sessão de treinamento do banco de dados.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Objeto TrainingSession
        """
        session = self._db.query(TrainingSession).get(session_id)
        if not session:
            raise ValueError(f"Sessão {session_id} não encontrada")
        return session

    async def delete_training_session(self, session_id: int) -> None:
        """
        Remove uma sessão de treinamento e seus arquivos.
        
        Args:
            session_id: ID da sessão
        """
        session = await self.get_training_session(session_id)
        
        # Remover diretório e arquivos
        session_dir = Path(session.log_file).parent
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Remover do banco
        self._db.delete(session)
        self._db.commit()

    async def list_training_sessions(
        self,
        dataset_id: Optional[int] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[TrainingSession]:
        """
        Lista sessões de treinamento do banco de dados.
        
        Args:
            dataset_id: ID do dataset (opcional)
            status: Status da sessão (opcional)
            skip: Número de registros para pular
            limit: Número máximo de registros
            
        Returns:
            Lista de objetos TrainingSession
        """
        query = self._db.query(TrainingSession)
        
        if dataset_id:
            query = query.filter(TrainingSession.dataset_id == dataset_id)
        if status:
            query = query.filter(TrainingSession.status == status)
        
        return query.order_by(TrainingSession.created_at.desc()).offset(skip).limit(limit).all()

    async def start_training(self, session_id: int) -> TrainingSession:
        """
        Inicia o treinamento de uma sessão de forma assíncrona em um processo separado.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Objeto TrainingSession atualizado
        """
        session = await self.get_training_session(session_id)
        
        # Atualizar status
        session.status = "training"
        session.started_at = datetime.utcnow()
        
        # Inicializar dados de progresso
        self._progress_data[session_id] = {
            "current_epoch": 0,
            "total_epochs": session.hyperparameters.get("epochs", 100),
            "metrics": {},
            "resources": {},
            "status": "running"
        }
        
        try:
            # Salvar configuração para o processo em um arquivo
            session_dir = Path(session.log_file).parent
            config_path = session_dir / "training_config.json"
            
            # Criar arquivo para comunicação entre processos
            progress_file = session_dir / "progress.json"
            with open(progress_file, "w") as f:
                json.dump(self._progress_data[session_id], f)
            
            # Verificar e corrigir o dispositivo se necessário
            hyperparameters = session.hyperparameters or {}
            if "device" in hyperparameters and hyperparameters["device"] == "auto" and not CUDA_AVAILABLE:
                hyperparameters = hyperparameters.copy()
                hyperparameters["device"] = "cpu"
                logger.info("CUDA não disponível. Forçando device=cpu para o treinamento.")
            elif "device" not in hyperparameters and not CUDA_AVAILABLE:
                hyperparameters = hyperparameters.copy()
                hyperparameters["device"] = "cpu"
                logger.info("CUDA não disponível. Forçando device=cpu para o treinamento.")
            
            config = {
                "session_id": session_id,
                "dataset_id": session.dataset_id,
                "model_type": session.model_type,
                "model_version": session.model_version,
                "hyperparameters": hyperparameters,
                "output_dir": str(session_dir),
                "progress_file": str(progress_file)  # Adicionar caminho do arquivo de progresso
            }
            
            with open(config_path, "w") as f:
                json.dump(config, f)
            
            # Criar processo para executar o treinamento
            cmd = [
                sys.executable,
                "-c",
                f"""
import json
import asyncio
import sys
import torch
from pathlib import Path

# Carregar configuração
config_path = "{config_path}"
with open(config_path, "r") as f:
    config = json.load(f)

# Configurar caminho no Python
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

# Importar serviços
from microdetect.services.yolo_service import YOLOService
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config["output_dir"]) / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("training_process")

# Verificar CUDA
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available in subprocess: {{cuda_available}}")

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

async def train_model():
    try:
        # Criar diretório de saída
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remover parâmetros que não são válidos para o YOLO
        hyperparameters = config["hyperparameters"].copy()
        for param in ["model_type", "model_size"]:
            if param in hyperparameters:
                hyperparameters.pop(param)
                
        # Preparar dataset e obter o caminho correto do data.yaml
        # Obter sessão do banco de dados
        db = next(get_db())
        
        from microdetect.services.dataset_service import DatasetService
        dataset_service = DatasetService(db)
        data_yaml_path = dataset_service.prepare_for_training(config["dataset_id"])
        logger.info(f"Dataset preparado. Usando arquivo data.yaml: {{data_yaml_path}}")
        
        # Treinar modelo
        logger.info(f"Starting training with params: {{hyperparameters}}")
        
        # Atualizar status inicial
        update_progress_file({{
            "total_epochs": hyperparameters.get("epochs", 100),
            "current_epoch": 0,
            "status": "running"
        }})
        
        # Definir callback para progresso
        def progress_callback(metrics):
            epoch = metrics.get("epoch", 0)
            logger.info(f"Epoch {{epoch}} completed. Metrics: {{metrics}}")
            
            # Atualizar progresso
            update_progress_file({{
                "current_epoch": epoch,
                "metrics": metrics
            }})
        
        # Inicializar YOLOService
        yolo_service = YOLOService()
        
        # Treinar modelo
        metrics = await yolo_service.train(
            dataset_id=config["dataset_id"],
            model_type=config["model_type"],
            model_version=config["model_version"],
            hyperparameters=hyperparameters,
            callback=progress_callback,
            db_session=db,
            data_yaml_path=data_yaml_path
        )
        
        # Atualizar status final
        update_progress_file({{
            "status": "completed",
            "metrics": metrics
        }})
        
        logger.info(f"Training completed successfully. Metrics: {{metrics}}")
        
        # Salvar resultados
        with open(output_dir / "results.json", "w") as f:
            json.dump(metrics, f)
            
    except Exception as e:
        logger.error(f"Error in training: {{str(e)}}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        # Atualizar status em caso de erro
        update_progress_file({{
            "status": "failed",
            "error": str(e)
        }})
    finally:
        # Fechar sessão do banco
        if 'db' in locals():
            db.close()

# Executar o treinamento
asyncio.run(train_model())
                """
            ]
            
            # Iniciar o processo
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Configurar monitoramento assíncrono do processo
            self._training_tasks[session_id] = {
                "process": process,
                "status": "running",
                "output_dir": session_dir,
                "progress_file": progress_file
            }
            
            # Iniciar tarefa para monitorar a conclusão
            asyncio.create_task(self._monitor_training_completion(session_id))
            
            # Iniciar tarefa para monitorar o arquivo de progresso
            asyncio.create_task(self._monitor_progress_file(session_id))
            
            self._db.commit()
            
            return session
            
        except Exception as e:
            # Atualizar status em caso de erro
            session.status = "failed"
            session.error_message = str(e)
            session.completed_at = datetime.utcnow()
            raise
    
    async def _monitor_progress_file(self, session_id: int):
        """
        Monitora o arquivo de progresso para atualizar os dados em tempo real.
        """
        if session_id not in self._training_tasks:
            return
        
        progress_file = self._training_tasks[session_id]["progress_file"]
        if not os.path.exists(progress_file):
            logger.error(f"Arquivo de progresso não encontrado: {progress_file}")
            return
        
        try:
            # Verificar o arquivo periodicamente
            while session_id in self._training_tasks and self._training_tasks[session_id]["status"] in ["running", "starting"]:
                try:
                    # Ler arquivo de progresso
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                    
                    # Atualizar dados de progresso
                    self.update_progress(session_id, progress_data)
                    
                    # Verificar se o treinamento terminou
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
    
    async def _monitor_training_completion(self, session_id: int):
        """
        Monitora a conclusão de uma tarefa de treinamento e atualiza o banco de dados.
        """
        if session_id not in self._training_tasks:
            return
            
        try:
            # Obter processo
            process = self._training_tasks[session_id]["process"]
            output_dir = Path(self._training_tasks[session_id]["output_dir"])
            
            # Esperar pelo processo
            stdout, stderr = await asyncio.get_event_loop().run_in_executor(
                None, process.communicate
            )
            
            # Registrar saída
            logger.info(f"Training process for {session_id} completed with code {process.returncode}")
            logger.debug(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
            
            if process.returncode != 0:
                logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                raise RuntimeError(f"Training process failed with code {process.returncode}")
            
            # Verificar se o arquivo de resultado existe
            results_file = output_dir / "results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found at {results_file}")
            
            # Carregar resultados
            with open(results_file, "r") as f:
                metrics = json.load(f)
            
            # Obter sessão do banco
            session = self._db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if not session:
                return
                
            # Atualizar resultados e status
            session.metrics = metrics
            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            self._db.commit()
            
            # Atualizar dados de progresso
            self.update_progress(session_id, {
                "status": "completed",
                "metrics": metrics
            })
            
            # Gerar relatório
            model_path = output_dir / "weights" / "best.pt"
            if model_path.exists():
                # Copiar modelo para pasta de modelos
                model_name = f"{session.model_type}{session.model_version}"
                target_path = settings.MODELS_DIR / f"{model_name}_{session.id}.pt"
                shutil.copy(model_path, target_path)
                
                # Criar modelo no banco
                model = Model(
                    name=f"Model from training {session_id}",
                    filepath=str(target_path),
                    training_session_id=session_id,
                    model_type=session.model_type,
                    model_version=session.model_version
                )
                self._db.add(model)
                self._db.commit()
            
            # Limpar tarefa
            self._training_tasks[session_id]["status"] = "completed"
            
        except Exception as e:
            # Obter sessão do banco
            session = self._db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if session:
                # Atualizar status em caso de erro
                session.status = TrainingStatus.FAILED
                session.error_message = str(e)
                session.completed_at = datetime.utcnow()
                self._db.commit()
            
            # Atualizar dados de progresso
            self.update_progress(session_id, {
                "status": "failed",
                "error": str(e)
            })
                
            # Limpar tarefa
            self._training_tasks[session_id]["status"] = "failed"
            logger.error(f"Error monitoring training completion: {str(e)}")
            
        finally:
            # Remover tarefa após um tempo
            await asyncio.sleep(60)  # Manter status por um minuto
            if session_id in self._training_tasks:
                del self._training_tasks[session_id]
            # Remover dados de progresso
            await asyncio.sleep(300)  # Manter status por mais tempo
            if session_id in self._progress_data:
                del self._progress_data[session_id]

    async def get_training_session_info(self, session_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre uma sessão de treinamento.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Dicionário com informações da sessão
        """
        session = await self.get_training_session(session_id)
        dataset = self._db.query(Dataset).get(session.dataset_id)
        
        return {
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "classes": dataset.classes
            },
            "model": {
                "type": session.model_type,
                "version": session.model_version
            },
            "hyperparameters": session.hyperparameters,
            "status": session.status,
            "metrics": session.metrics,
            "error_message": session.error_message,
            "created_at": session.created_at,
            "started_at": session.started_at,
            "completed_at": session.completed_at
        }

    async def get_training_log(self, session_id: int) -> str:
        """
        Obtém o log de treinamento de uma sessão.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Conteúdo do arquivo de log
        """
        session = await self.get_training_session(session_id)
        
        if not os.path.exists(session.log_file):
            return ""
        
        with open(session.log_file, "r") as f:
            return f.read()

    async def export_model(
        self,
        session_id: int,
        format: str = "onnx"
    ) -> str:
        """
        Exporta o modelo treinado em uma sessão.
        
        Args:
            session_id: ID da sessão
            format: Formato de exportação (onnx, torchscript, etc.)
            
        Returns:
            Caminho do modelo exportado
        """
        session = await self.get_training_session(session_id)
        
        if session.status != "completed":
            raise ValueError("Sessão não concluída")
        
        # Exportar modelo
        export_path = await self.yolo_service.export(
            model_id=session.model_id,
            format=format
        )
        
        return export_path

    def __del__(self):
        """Fecha a sessão do banco quando o serviço for destruído."""
        try:
            self._db.close()
        except:
            pass

    async def train_model(self, session_id: int, db: Session) -> Dict[str, Any]:
        """
        Treina um modelo com base na sessão de treinamento.
        
        Args:
            session_id: ID da sessão de treinamento
            db: Sessão do banco de dados
            
        Returns:
            Métricas de treinamento
        """
        # Obter sessão de treinamento
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        if session.status not in [TrainingStatus.PENDING, TrainingStatus.FAILED]:
            logger.warning(f"Session {session_id} is already {session.status}")
            return session.metrics or {}
            
        # Atualizar status
        session.status = TrainingStatus.RUNNING
        session.started_at = datetime.utcnow()
        db.commit()
        
        try:
            # Configurar diretório de saída
            model_dir = settings.TRAINING_DIR / f"model_{session.id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar parâmetros de treinamento
            hyperparameters = session.hyperparameters or {}
            hyperparameters["project"] = str(model_dir.parent)
            hyperparameters["name"] = model_dir.name
            hyperparameters["exist_ok"] = True
            
            # Garantir que os parâmetros estejam no formato correto
            if "batch_size" in hyperparameters:
                hyperparameters["batch"] = hyperparameters.pop("batch_size")
                
            # Remover parâmetros que não são válidos para o YOLO
            for param in ["model_type", "model_size"]:
                if param in hyperparameters:
                    hyperparameters.pop(param)
                    
            # Preparar dataset e obter o caminho correto do data.yaml
            dataset_service = DatasetService(db)
            data_yaml_path = dataset_service.prepare_for_training(session.dataset_id)
            logger.info(f"Dataset preparado. Usando arquivo data.yaml: {data_yaml_path}")
                    
            # Treinar modelo com progresso em tempo real
            metrics = await self.yolo_service.train(
                dataset_id=session.dataset_id,
                model_type=session.model_type,
                model_version=session.model_version,
                hyperparameters=hyperparameters,
                callback=lambda metrics: self.update_progress(session_id, metrics, db),
                db_session=db,  # Passar a sessão do banco de dados
                data_yaml_path=data_yaml_path  # Passar o caminho do data.yaml
            )
            
            # Atualizar métricas e status
            session.metrics = metrics
            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            
            # Copiar modelo para pasta de modelos
            model_path = model_dir / "weights" / "best.pt"
            if model_path.exists():
                model_name = f"{session.model_type}{session.model_version}"
                model_filename = f"{model_name}_{session.id}.pt"
                target_path = settings.MODELS_DIR / model_filename
                shutil.copy(model_path, target_path)
                
                # Atualizar o caminho do modelo na sessão
                session.model_path = str(target_path)
                
                # Criar objeto de modelo
                model = Model(
                    name=f"{session.name} - {model_name}",
                    description=session.description,
                    filepath=str(target_path),
                    model_type=session.model_type,
                    model_version=session.model_version,
                    metrics=metrics,
                    training_session_id=session.id
                )
                db.add(model)
            
        except Exception as e:
            logger.error(f"Error in training session {session_id}: {e}")
            session.status = TrainingStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.utcnow()
        finally:
            # Salvar alterações
            db.commit()
            
        return session.metrics or {}

    def get_progress(self, session_id: int) -> Dict[str, Any]:
        """
        Obtém os dados de progresso em tempo real de uma sessão.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Dados de progresso da sessão
        """
        if session_id not in self._progress_data:
            return {
                "current_epoch": 0,
                "total_epochs": 0,
                "metrics": {},
                "resources": {},
                "status": "pending"
            }
        return self._progress_data[session_id]
    
    def update_progress(self, session_id: int, progress_data: Dict[str, Any]):
        """
        Atualiza os dados de progresso em tempo real de uma sessão.
        
        Args:
            session_id: ID da sessão
            progress_data: Novos dados de progresso
        """
        if session_id not in self._progress_data:
            self._progress_data[session_id] = {
                "current_epoch": 0,
                "total_epochs": 0,
                "metrics": {},
                "resources": {},
                "status": "pending"
            }
        
        # Atualizar com novos dados
        self._progress_data[session_id].update(progress_data)
        
        # Atualizar no banco de dados se tiver dados significativos
        try:
            if progress_data.get("metrics") and "map50" in progress_data.get("metrics", {}):
                session = self._db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
                if session:
                    session.metrics = progress_data.get("metrics", {})
                    self._db.commit()
        except Exception as e:
            logger.error(f"Erro ao atualizar progresso no banco: {e}") 
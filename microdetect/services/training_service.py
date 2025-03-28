import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.training_session import TrainingSession
from microdetect.models.model import Model
from microdetect.models.dataset import Dataset
from microdetect.services.yolo_service import YOLOService

class TrainingService:
    def __init__(self):
        self.training_dir = settings.TRAINING_DIR
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_service = YOLOService()

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
        dataset = Dataset.query.get(dataset_id)
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
        session = TrainingSession.query.get(session_id)
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
        session.delete()

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
        query = TrainingSession.query
        
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        if status:
            query = query.filter_by(status=status)
        
        return query.order_by(TrainingSession.created_at.desc()).offset(skip).limit(limit).all()

    async def start_training(self, session_id: int) -> TrainingSession:
        """
        Inicia o treinamento de uma sessão.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Objeto TrainingSession atualizado
        """
        session = await self.get_training_session(session_id)
        
        # Atualizar status
        session.status = "training"
        session.started_at = datetime.utcnow()
        
        try:
            # Treinar modelo
            metrics = await self.yolo_service.train(
                dataset_id=session.dataset_id,
                model_type=session.model_type,
                model_version=session.model_version,
                hyperparameters=session.hyperparameters
            )
            
            # Atualizar métricas
            session.metrics = metrics
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            
            # Criar modelo treinado
            model = Model(
                name=f"{session.name} - {session.model_type}{session.model_version}",
                description=session.description,
                filepath=str(Path(session.log_file).parent / "weights" / "best.pt"),
                model_type=session.model_type,
                model_version=session.model_version,
                metrics=metrics
            )
            
        except Exception as e:
            # Atualizar status em caso de erro
            session.status = "failed"
            session.error_message = str(e)
            session.completed_at = datetime.utcnow()
            raise
        
        return session

    async def get_training_session_info(self, session_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre uma sessão de treinamento.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Dicionário com informações da sessão
        """
        session = await self.get_training_session(session_id)
        dataset = Dataset.query.get(session.dataset_id)
        
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
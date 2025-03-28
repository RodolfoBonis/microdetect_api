import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.model import Model
from microdetect.services.yolo_service import YOLOService

class ModelService:
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_service = YOLOService()

    async def create_model(
        self,
        name: str,
        description: Optional[str] = None,
        model_type: str = "yolov8",
        model_version: str = "n",
        filepath: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Model:
        """
        Cria um novo modelo.
        
        Args:
            name: Nome do modelo
            description: Descrição do modelo (opcional)
            model_type: Tipo do modelo (ex: "yolov8")
            model_version: Versão do modelo (ex: "n", "s", "m", "l", "x")
            filepath: Caminho do arquivo do modelo (opcional)
            metrics: Métricas do modelo (opcional)
            
        Returns:
            Objeto Model criado
        """
        # Criar diretório do modelo
        model_dir = self.models_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir.mkdir(exist_ok=True)
        
        # Se um arquivo foi fornecido, copiá-lo
        if filepath and os.path.exists(filepath):
            shutil.copy2(filepath, model_dir / "model.pt")
            filepath = str(model_dir / "model.pt")
        
        # Criar registro no banco
        model = Model(
            name=name,
            description=description,
            model_type=model_type,
            model_version=model_version,
            filepath=filepath or str(model_dir / "model.pt"),
            metrics=metrics or {}
        )
        
        # Salvar metadados
        metadata = {
            "name": model.name,
            "description": model.description,
            "model_type": model.model_type,
            "model_version": model.model_version,
            "metrics": model.metrics,
            "created_at": model.created_at.isoformat()
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        return model

    async def get_model(self, model_id: int) -> Model:
        """
        Recupera um modelo do banco de dados.
        
        Args:
            model_id: ID do modelo
            
        Returns:
            Objeto Model
        """
        model = Model.query.get(model_id)
        if not model:
            raise ValueError(f"Modelo {model_id} não encontrado")
        return model

    async def delete_model(self, model_id: int) -> None:
        """
        Remove um modelo e seus arquivos.
        
        Args:
            model_id: ID do modelo
        """
        model = await self.get_model(model_id)
        
        # Remover diretório e arquivos
        model_dir = Path(model.filepath).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remover do banco
        model.delete()

    async def list_models(
        self,
        model_type: Optional[str] = None,
        model_version: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Model]:
        """
        Lista modelos do banco de dados.
        
        Args:
            model_type: Tipo do modelo (opcional)
            model_version: Versão do modelo (opcional)
            skip: Número de registros para pular
            limit: Número máximo de registros
            
        Returns:
            Lista de objetos Model
        """
        query = Model.query
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        if model_version:
            query = query.filter_by(model_version=model_version)
        
        return query.order_by(Model.created_at.desc()).offset(skip).limit(limit).all()

    async def update_model(
        self,
        model_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Model:
        """
        Atualiza um modelo.
        
        Args:
            model_id: ID do modelo
            name: Novo nome (opcional)
            description: Nova descrição (opcional)
            metrics: Novas métricas (opcional)
            
        Returns:
            Objeto Model atualizado
        """
        model = await self.get_model(model_id)
        
        # Atualizar campos
        if name:
            model.name = name
        if description is not None:
            model.description = description
        if metrics is not None:
            model.metrics = metrics
        
        # Atualizar metadados
        model_dir = Path(model.filepath).parent
        metadata_file = model_dir / "metadata.json"
        
        if metadata_file.exists():
            metadata = {
                "name": model.name,
                "description": model.description,
                "model_type": model.model_type,
                "model_version": model.model_version,
                "metrics": model.metrics,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat()
            }
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
        
        return model

    async def get_model_info(self, model_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre um modelo.
        
        Args:
            model_id: ID do modelo
            
        Returns:
            Dicionário com informações do modelo
        """
        model = await self.get_model(model_id)
        
        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "model_type": model.model_type,
            "model_version": model.model_version,
            "filepath": model.filepath,
            "metrics": model.metrics,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "training_sessions": [
                {
                    "id": session.id,
                    "name": session.name,
                    "status": session.status,
                    "metrics": session.metrics
                }
                for session in model.training_sessions
            ]
        }

    async def validate_model(
        self,
        model_id: int,
        dataset_id: int
    ) -> Dict[str, Any]:
        """
        Valida um modelo em um dataset.
        
        Args:
            model_id: ID do modelo
            dataset_id: ID do dataset
            
        Returns:
            Dicionário com métricas de validação
        """
        model = await self.get_model(model_id)
        
        # Validar modelo
        metrics = await self.yolo_service.validate(
            model_id=model_id,
            dataset_id=dataset_id
        )
        
        # Atualizar métricas do modelo
        model.metrics.update(metrics)
        
        return metrics

    async def export_model(
        self,
        model_id: int,
        format: str = "onnx"
    ) -> str:
        """
        Exporta um modelo para outro formato.
        
        Args:
            model_id: ID do modelo
            format: Formato de exportação (onnx, torchscript, etc.)
            
        Returns:
            Caminho do modelo exportado
        """
        model = await self.get_model(model_id)
        
        # Exportar modelo
        export_path = await self.yolo_service.export(
            model_id=model_id,
            format=format
        )
        
        return export_path

    async def get_model_versions(self, model_type: str) -> List[str]:
        """
        Obtém as versões disponíveis para um tipo de modelo.
        
        Args:
            model_type: Tipo do modelo
            
        Returns:
            Lista de versões disponíveis
        """
        if model_type == "yolov8":
            return ["n", "s", "m", "l", "x"]
        return []

    async def get_model_metrics(self, model_id: int) -> Dict[str, Any]:
        """
        Obtém as métricas detalhadas de um modelo.
        
        Args:
            model_id: ID do modelo
            
        Returns:
            Dicionário com métricas detalhadas
        """
        model = await self.get_model(model_id)
        
        # Coletar métricas de todas as sessões de treinamento
        training_metrics = []
        for session in model.training_sessions:
            if session.status == "completed":
                training_metrics.append({
                    "session_id": session.id,
                    "name": session.name,
                    "metrics": session.metrics,
                    "created_at": session.created_at
                })
        
        # Coletar métricas de validação
        validation_metrics = []
        for session in model.training_sessions:
            if session.status == "completed":
                validation_metrics.append({
                    "session_id": session.id,
                    "name": session.name,
                    "metrics": session.metrics,
                    "created_at": session.created_at
                })
        
        return {
            "model_metrics": model.metrics,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics
        } 
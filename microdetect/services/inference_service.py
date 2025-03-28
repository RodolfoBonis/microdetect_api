import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.inference_result import InferenceResult
from microdetect.models.model import Model
from microdetect.models.image import Image
from microdetect.services.yolo_service import YOLOService
from microdetect.services.image_service import ImageService

class InferenceService:
    def __init__(self):
        self.inference_dir = settings.INFERENCE_DIR
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_service = YOLOService()
        self.image_service = ImageService()

    async def perform_inference(
        self,
        model_id: int,
        image_id: int,
        confidence_threshold: float = 0.5
    ) -> InferenceResult:
        """
        Realiza inferência em uma imagem usando um modelo.
        
        Args:
            model_id: ID do modelo
            image_id: ID da imagem
            confidence_threshold: Limiar de confiança
            
        Returns:
            Objeto InferenceResult criado
        """
        # Verificar modelo
        model = Model.query.get(model_id)
        if not model:
            raise ValueError(f"Modelo {model_id} não encontrado")
        
        # Verificar imagem
        image = Image.query.get(image_id)
        if not image:
            raise ValueError(f"Imagem {image_id} não encontrada")
        
        # Realizar inferência
        predictions, metrics = await self.yolo_service.predict(
            model_id=model_id,
            image_path=image.filepath,
            confidence_threshold=confidence_threshold
        )
        
        # Criar diretório do resultado
        result_dir = self.inference_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_dir.mkdir(exist_ok=True)
        
        # Salvar resultado
        result_file = result_dir / "result.json"
        result_data = {
            "image": {
                "id": image.id,
                "filename": image.filename,
                "width": image.width,
                "height": image.height
            },
            "model": {
                "id": model.id,
                "name": model.name,
                "type": model.model_type,
                "version": model.model_version
            },
            "predictions": predictions,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(result_file, "w") as f:
            json.dump(result_data, f)
        
        # Criar registro no banco
        result = InferenceResult(
            image_id=image_id,
            model_id=model_id,
            predictions=predictions,
            metrics=metrics,
            filepath=str(result_file)
        )
        
        return result

    async def get_inference_result(self, result_id: int) -> InferenceResult:
        """
        Recupera um resultado de inferência do banco de dados.
        
        Args:
            result_id: ID do resultado
            
        Returns:
            Objeto InferenceResult
        """
        result = InferenceResult.query.get(result_id)
        if not result:
            raise ValueError(f"Resultado {result_id} não encontrado")
        return result

    async def delete_inference_result(self, result_id: int) -> None:
        """
        Remove um resultado de inferência e seus arquivos.
        
        Args:
            result_id: ID do resultado
        """
        result = await self.get_inference_result(result_id)
        
        # Remover diretório e arquivos
        result_dir = Path(result.filepath).parent
        if result_dir.exists():
            shutil.rmtree(result_dir)
        
        # Remover do banco
        result.delete()

    async def list_inference_results(
        self,
        image_id: Optional[int] = None,
        model_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[InferenceResult]:
        """
        Lista resultados de inferência do banco de dados.
        
        Args:
            image_id: ID da imagem (opcional)
            model_id: ID do modelo (opcional)
            skip: Número de registros para pular
            limit: Número máximo de registros
            
        Returns:
            Lista de objetos InferenceResult
        """
        query = InferenceResult.query
        
        if image_id:
            query = query.filter_by(image_id=image_id)
        if model_id:
            query = query.filter_by(model_id=model_id)
        
        return query.order_by(InferenceResult.created_at.desc()).offset(skip).limit(limit).all()

    async def get_inference_result_info(self, result_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre um resultado de inferência.
        
        Args:
            result_id: ID do resultado
            
        Returns:
            Dicionário com informações do resultado
        """
        result = await self.get_inference_result(result_id)
        image = Image.query.get(result.image_id)
        model = Model.query.get(result.model_id)
        
        return {
            "id": result.id,
            "image": {
                "id": image.id,
                "filename": image.filename,
                "width": image.width,
                "height": image.height
            },
            "model": {
                "id": model.id,
                "name": model.name,
                "type": model.model_type,
                "version": model.model_version
            },
            "predictions": result.predictions,
            "metrics": result.metrics,
            "created_at": result.created_at
        }

    async def batch_inference(
        self,
        model_id: int,
        image_ids: List[int],
        confidence_threshold: float = 0.5
    ) -> List[InferenceResult]:
        """
        Realiza inferência em várias imagens.
        
        Args:
            model_id: ID do modelo
            image_ids: Lista de IDs de imagens
            confidence_threshold: Limiar de confiança
            
        Returns:
            Lista de objetos InferenceResult
        """
        results = []
        
        for image_id in image_ids:
            try:
                result = await self.perform_inference(
                    model_id=model_id,
                    image_id=image_id,
                    confidence_threshold=confidence_threshold
                )
                results.append(result)
            except Exception as e:
                # Log do erro e continuar com as próximas imagens
                print(f"Erro ao processar imagem {image_id}: {str(e)}")
                continue
        
        return results

    async def export_inference_results(
        self,
        result_ids: List[int],
        format: str = "json"
    ) -> str:
        """
        Exporta resultados de inferência em um formato específico.
        
        Args:
            result_ids: Lista de IDs de resultados
            format: Formato de exportação (json, csv, etc.)
            
        Returns:
            Caminho do arquivo exportado
        """
        # Criar diretório de exportação
        export_dir = self.inference_dir / "exports" / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(exist_ok=True)
        
        if format == "json":
            # Coletar resultados
            results = []
            for result_id in result_ids:
                result = await self.get_inference_result(result_id)
                results.append(await self.get_inference_result_info(result_id))
            
            # Salvar arquivo JSON
            with open(export_dir / "results.json", "w") as f:
                json.dump(results, f)
        
        elif format == "csv":
            import csv
            
            # Coletar resultados
            results = []
            for result_id in result_ids:
                result = await self.get_inference_result(result_id)
                info = await self.get_inference_result_info(result_id)
                
                # Extrair métricas
                metrics = info["metrics"]
                
                # Criar linha do CSV
                row = {
                    "image_id": info["image"]["id"],
                    "image_filename": info["image"]["filename"],
                    "model_id": info["model"]["id"],
                    "model_name": info["model"]["name"],
                    "num_predictions": len(info["predictions"]),
                    "inference_time": metrics.get("inference_time", 0),
                    "fps": metrics.get("fps", 0),
                    "created_at": info["created_at"]
                }
                results.append(row)
            
            # Salvar arquivo CSV
            with open(export_dir / "results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        else:
            raise ValueError(f"Formato de exportação não suportado: {format}")
        
        return str(export_dir) 
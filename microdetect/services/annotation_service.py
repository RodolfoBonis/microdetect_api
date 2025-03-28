import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from microdetect.core.config import settings
from microdetect.models.annotation import Annotation
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset

class AnnotationService:
    def __init__(self):
        self.annotations_dir = settings.ANNOTATIONS_DIR
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    async def create_annotation(
        self,
        image_id: int,
        bbox: List[float],
        class_id: int,
        confidence: Optional[float] = None
    ) -> Annotation:
        """
        Cria uma nova anotação.
        
        Args:
            image_id: ID da imagem
            bbox: Lista com coordenadas [x1, y1, x2, y2]
            class_id: ID da classe
            confidence: Confiança da anotação (opcional)
            
        Returns:
            Objeto Annotation criado
        """
        # Verificar imagem
        image = Image.query.get(image_id)
        if not image:
            raise ValueError(f"Imagem {image_id} não encontrada")
        
        # Verificar classe
        dataset = Dataset.query.get(image.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {image.dataset_id} não encontrado")
        
        if class_id >= len(dataset.classes):
            raise ValueError(f"Classe {class_id} não encontrada no dataset")
        
        # Criar diretório da imagem se não existir
        image_dir = self.annotations_dir / str(image_id)
        image_dir.mkdir(exist_ok=True)
        
        # Gerar nome do arquivo
        annotation_count = len(image.annotations)
        filename = f"annotation_{annotation_count + 1}.json"
        filepath = image_dir / filename
        
        # Criar dados da anotação
        annotation_data = {
            "bbox": bbox,
            "class_id": class_id,
            "class_name": dataset.classes[class_id],
            "confidence": confidence
        }
        
        # Salvar arquivo
        with open(filepath, "w") as f:
            json.dump(annotation_data, f)
        
        # Criar registro no banco
        annotation = Annotation(
            filename=filename,
            filepath=str(filepath),
            image_id=image_id,
            class_id=class_id,
            confidence=confidence
        )
        
        return annotation

    async def get_annotation(self, annotation_id: int) -> Annotation:
        """
        Recupera uma anotação do banco de dados.
        
        Args:
            annotation_id: ID da anotação
            
        Returns:
            Objeto Annotation
        """
        annotation = Annotation.query.get(annotation_id)
        if not annotation:
            raise ValueError(f"Anotação {annotation_id} não encontrada")
        return annotation

    async def delete_annotation(self, annotation_id: int) -> None:
        """
        Remove uma anotação do sistema de arquivos e do banco de dados.
        
        Args:
            annotation_id: ID da anotação
        """
        annotation = await self.get_annotation(annotation_id)
        
        # Remover arquivo
        if os.path.exists(annotation.filepath):
            os.remove(annotation.filepath)
        
        # Remover do banco
        annotation.delete()

    async def list_annotations(
        self,
        image_id: Optional[int] = None,
        class_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Annotation]:
        """
        Lista anotações do banco de dados.
        
        Args:
            image_id: ID da imagem (opcional)
            class_id: ID da classe (opcional)
            skip: Número de registros para pular
            limit: Número máximo de registros
            
        Returns:
            Lista de objetos Annotation
        """
        query = Annotation.query
        
        if image_id:
            query = query.filter_by(image_id=image_id)
        if class_id is not None:
            query = query.filter_by(class_id=class_id)
        
        return query.offset(skip).limit(limit).all()

    async def update_annotation(
        self,
        annotation_id: int,
        bbox: Optional[List[float]] = None,
        class_id: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> Annotation:
        """
        Atualiza uma anotação.
        
        Args:
            annotation_id: ID da anotação
            bbox: Nova lista de coordenadas [x1, y1, x2, y2] (opcional)
            class_id: Novo ID da classe (opcional)
            confidence: Nova confiança (opcional)
            
        Returns:
            Objeto Annotation atualizado
        """
        annotation = await self.get_annotation(annotation_id)
        image = Image.query.get(annotation.image_id)
        dataset = Dataset.query.get(image.dataset_id)
        
        # Verificar classe se fornecida
        if class_id is not None and class_id >= len(dataset.classes):
            raise ValueError(f"Classe {class_id} não encontrada no dataset")
        
        # Atualizar campos
        if bbox is not None:
            annotation.bbox = bbox
        if class_id is not None:
            annotation.class_id = class_id
        if confidence is not None:
            annotation.confidence = confidence
        
        # Atualizar arquivo
        annotation_data = {
            "bbox": annotation.bbox,
            "class_id": annotation.class_id,
            "class_name": dataset.classes[annotation.class_id],
            "confidence": annotation.confidence
        }
        
        with open(annotation.filepath, "w") as f:
            json.dump(annotation_data, f)
        
        return annotation

    async def get_annotation_info(self, annotation_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre uma anotação.
        
        Args:
            annotation_id: ID da anotação
            
        Returns:
            Dicionário com informações da anotação
        """
        annotation = await self.get_annotation(annotation_id)
        image = Image.query.get(annotation.image_id)
        dataset = Dataset.query.get(image.dataset_id)
        
        return {
            "id": annotation.id,
            "filename": annotation.filename,
            "filepath": annotation.filepath,
            "image_id": annotation.image_id,
            "class_id": annotation.class_id,
            "class_name": dataset.classes[annotation.class_id],
            "confidence": annotation.confidence,
            "created_at": annotation.created_at,
            "updated_at": annotation.updated_at
        }

    async def export_annotations(
        self,
        dataset_id: int,
        format: str = "yolo"
    ) -> str:
        """
        Exporta anotações de um dataset em um formato específico.
        
        Args:
            dataset_id: ID do dataset
            format: Formato de exportação (yolo, coco, etc.)
            
        Returns:
            Caminho do arquivo exportado
        """
        dataset = await Dataset.query.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Criar diretório de exportação
        export_dir = self.annotations_dir / "exports" / str(dataset_id)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "yolo":
            # Criar diretório de labels
            labels_dir = export_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Processar cada imagem
            for image in Image.query.filter_by(dataset_id=dataset_id).all():
                # Criar arquivo de labels
                label_file = labels_dir / f"{Path(image.filename).stem}.txt"
                
                with open(label_file, "w") as f:
                    for annotation in image.annotations:
                        # Converter bbox para formato YOLO [x_center, y_center, width, height]
                        x1, y1, x2, y2 = annotation.bbox
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Normalizar coordenadas
                        x_center /= image.width
                        y_center /= image.height
                        width /= image.width
                        height /= image.height
                        
                        # Escrever linha no arquivo
                        f.write(f"{annotation.class_id} {x_center} {y_center} {width} {height}\n")
        
        elif format == "coco":
            # Criar estrutura COCO
            coco_data = {
                "info": {
                    "description": dataset.description or "",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "MicroDetect",
                    "date_created": dataset.created_at.isoformat()
                },
                "licenses": [],
                "categories": [
                    {"id": i, "name": name, "supercategory": "none"}
                    for i, name in enumerate(dataset.classes)
                ],
                "images": [],
                "annotations": []
            }
            
            # Processar cada imagem
            for image in Image.query.filter_by(dataset_id=dataset_id).all():
                # Adicionar imagem
                image_data = {
                    "id": image.id,
                    "file_name": image.filename,
                    "height": image.height,
                    "width": image.width,
                    "date_captured": image.created_at.isoformat()
                }
                coco_data["images"].append(image_data)
                
                # Adicionar anotações
                for annotation in image.annotations:
                    x1, y1, x2, y2 = annotation.bbox
                    annotation_data = {
                        "id": annotation.id,
                        "image_id": image.id,
                        "category_id": annotation.class_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_data)
            
            # Salvar arquivo COCO
            with open(export_dir / "annotations.json", "w") as f:
                json.dump(coco_data, f)
        
        else:
            raise ValueError(f"Formato de exportação não suportado: {format}")
        
        return str(export_dir) 
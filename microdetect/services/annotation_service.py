import os
import json
import shutil
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
        self.images_dir = settings.IMAGES_DIR
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = settings.TRAINING_DIR
        self.training_dir.mkdir(parents=True, exist_ok=True)

    async def create_annotation(
        self,
        image_id: int,
        bbox: Dict[str, float],
        class_name: str,
        confidence: Optional[float] = None
    ) -> Annotation:
        """
        Cria uma nova anotação.
        
        Args:
            image_id: ID da imagem
            bbox: Dicionário com coordenadas {x, y, width, height}
            class_name: Nome da classe
            confidence: Confiança da anotação (opcional)
            
        Returns:
            Objeto Annotation criado
        """
        # Verificar imagem
        image = Image.query.get(image_id)
        if not image:
            raise ValueError(f"Imagem {image_id} não encontrada")
        
        # Verificar dataset
        dataset_id = image.dataset_id
        if dataset_id:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} não encontrado")
            
            # Adicionar classe ao dataset se não existir
            if class_name and dataset.classes and class_name not in dataset.classes:
                classes = dataset.classes.copy()
                classes.append(class_name)
                dataset.classes = classes
        
        # Preparar dados do bounding box
        if isinstance(bbox, dict):
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
        elif isinstance(bbox, list) and len(bbox) == 4:
            # [x1, y1, x2, y2]
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y
        else:
            raise ValueError("Formato de bounding box inválido")
        
        # Criar diretório de anotações para o dataset se não existir
        if dataset_id:
            dataset_annotations_dir = self.annotations_dir / str(dataset_id)
            dataset_annotations_dir.mkdir(exist_ok=True)
        
        # Criar anotação
        annotation = Annotation(
            image_id=image_id,
            dataset_id=dataset_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            x=x,
            y=y,
            width=width,
            height=height,
            area=width * height
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
        export_format: str = "yolo",
        destination_dir: Optional[Path] = None
    ) -> str:
        """
        Exporta anotações de um dataset para treinamento YOLO.
        
        Args:
            dataset_id: ID do dataset
            export_format: Formato de exportação (yolo, coco, etc.)
            destination_dir: Diretório de destino personalizado (opcional)
            
        Returns:
            Caminho do diretório de exportação
        """
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Criar diretório de exportação
        if destination_dir:
            export_dir = destination_dir
        else:
            # Usar o nome do dataset ou ID se nome for None
            dataset_name = dataset.name if dataset.name else f"dataset_{dataset_id}"
            export_dir = self.training_dir / dataset_name
        
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format.lower() == "yolo":
            # Estrutura de diretórios para YOLO
            images_dir = export_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            labels_dir = export_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Criar subpastas para treino, validação e teste
            for split in ["train", "val", "test"]:
                (images_dir / split).mkdir(exist_ok=True)
                (labels_dir / split).mkdir(exist_ok=True)
            
            # Criar arquivo classes.txt
            with open(export_dir / "classes.txt", "w") as f:
                for cls in dataset.classes:
                    f.write(f"{cls}\n")
            
            # Criar arquivo data.yaml
            data_yaml = {
                "path": str(export_dir.absolute()),
                "train": str((images_dir / "train").absolute()),
                "val": str((images_dir / "val").absolute()),
                "test": str((images_dir / "test").absolute()),
                "names": {i: name for i, name in enumerate(dataset.classes)},
                "nc": len(dataset.classes)
            }
            
            with open(export_dir / "data.yaml", "w") as f:
                import yaml
                yaml.dump(data_yaml, f, sort_keys=False)
            
            # Processar cada imagem do dataset
            images = Image.query.filter_by(dataset_id=dataset_id).all()
            
            # Dividir em treino (70%), validação (20%) e teste (10%)
            import random
            random.shuffle(images)
            
            train_idx = int(len(images) * 0.7)
            val_idx = int(len(images) * 0.9)
            
            train_images = images[:train_idx]
            val_images = images[train_idx:val_idx]
            test_images = images[val_idx:]
            
            # Processar imagens de treino
            await self._process_images_for_split(train_images, "train", images_dir, labels_dir, dataset.classes)
            
            # Processar imagens de validação
            await self._process_images_for_split(val_images, "val", images_dir, labels_dir, dataset.classes)
            
            # Processar imagens de teste
            await self._process_images_for_split(test_images, "test", images_dir, labels_dir, dataset.classes)
            
        elif export_format.lower() == "coco":
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
            raise ValueError(f"Formato de exportação não suportado: {export_format}")
        
        return str(export_dir)
    
    async def _process_images_for_split(
        self, 
        images: List[Image], 
        split: str, 
        images_dir: Path, 
        labels_dir: Path, 
        classes: List[str]
    ):
        """
        Processa imagens para um determinado split (train/val/test)
        
        Args:
            images: Lista de objetos Image
            split: Nome do split (train, val, test)
            images_dir: Diretório base de imagens
            labels_dir: Diretório base de labels
            classes: Lista de classes do dataset
        """
        for image in images:
            # Copiar imagem para pasta do split
            dest_images_dir = images_dir / split
            image_dest = dest_images_dir / image.file_name
            
            # Copiar a imagem se ela existir
            if os.path.exists(image.file_path):
                shutil.copy(image.file_path, image_dest)
            
            # Criar arquivo de anotações
            dest_labels_dir = labels_dir / split
            label_file = dest_labels_dir / f"{Path(image.file_name).stem}.txt"
            await self._create_yolo_annotation_file(image, label_file, classes)
    
    async def _create_yolo_annotation_file(self, image: Image, label_file: Path, classes: List[str]) -> None:
        """
        Cria um arquivo de anotação no formato YOLO para uma imagem.
        
        Args:
            image: Objeto Image
            label_file: Caminho do arquivo de saída
            classes: Lista de classes do dataset
        """
        with open(label_file, "w") as f:
            for annotation in image.annotations:
                # Obter o índice da classe
                class_idx = -1
                if annotation.class_name in classes:
                    class_idx = classes.index(annotation.class_name)
                else:
                    # Se a classe não estiver na lista, pular esta anotação
                    continue
                
                # Extrair coordenadas do bounding box
                # Se estiver usando bbox como JSON
                if hasattr(annotation, 'bbox') and annotation.bbox:
                    if isinstance(annotation.bbox, dict):
                        x = annotation.bbox.get('x', 0)
                        y = annotation.bbox.get('y', 0)
                        w = annotation.bbox.get('width', 0)
                        h = annotation.bbox.get('height', 0)
                    elif isinstance(annotation.bbox, list) and len(annotation.bbox) == 4:
                        # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = annotation.bbox
                        w = x2 - x1
                        h = y2 - y1
                        x = x1
                        y = y1
                # Se estiver usando campos separados
                elif all(hasattr(annotation, attr) for attr in ['x', 'y', 'width', 'height']):
                    x = annotation.x
                    y = annotation.y
                    w = annotation.width
                    h = annotation.height
                else:
                    # Se não tiver as informações necessárias, pular
                    continue
                
                # Converter para formato YOLO (x_center, y_center, width, height normalizado)
                x_center = (x + w/2) / image.width
                y_center = (y + h/2) / image.height
                width = w / image.width
                height = h / image.height
                
                # Escrever no formato YOLO: class_idx x_center y_center width height
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    async def import_annotations(
        self,
        dataset_id: int, 
        import_format: str, 
        source_dir: Path
    ) -> int:
        """
        Importa anotações de um diretório para o banco de dados.
        
        Args:
            dataset_id: ID do dataset
            import_format: Formato das anotações (yolo, coco)
            source_dir: Diretório com as anotações
            
        Returns:
            Número de anotações importadas
        """
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        count = 0
        
        if import_format.lower() == "yolo":
            # Verificar arquivo de classes
            classes_file = source_dir / "classes.txt"
            if classes_file.exists():
                with open(classes_file, "r") as f:
                    classes = [line.strip() for line in f.readlines()]
                
                # Atualizar classes do dataset
                dataset.classes = classes
                
            # Localizar arquivos de anotação
            labels_dir = source_dir / "labels"
            if not labels_dir.exists():
                # Tentar encontrar labels em train/val
                labels_dirs = [(source_dir / "labels" / "train"), (source_dir / "labels" / "val")]
                labels_dirs = [d for d in labels_dirs if d.exists()]
            else:
                labels_dirs = [labels_dir]
            
            if not labels_dirs:
                raise ValueError(f"Nenhum diretório de anotações encontrado em {source_dir}")
            
            # Processar cada arquivo de anotação
            for labels_dir in labels_dirs:
                for label_file in labels_dir.glob("*.txt"):
                    # Encontrar imagem correspondente
                    image_name = label_file.stem + ".jpg"  # tentar jpg primeiro
                    image = Image.query.filter_by(dataset_id=dataset_id, file_name=image_name).first()
                    
                    if not image:
                        # Tentar com extensão png
                        image_name = label_file.stem + ".png"
                        image = Image.query.filter_by(dataset_id=dataset_id, file_name=image_name).first()
                    
                    if not image:
                        # Ignorar se não encontrar a imagem
                        continue
                    
                    # Importar anotações
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:  # class_idx x_center y_center width height
                                class_idx = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Converter para coordenadas absolutas
                                x = (x_center - width/2) * image.width
                                y = (y_center - height/2) * image.height
                                w = width * image.width
                                h = height * image.height
                                
                                # Criar anotação
                                class_name = dataset.classes[class_idx] if class_idx < len(dataset.classes) else f"class_{class_idx}"
                                annotation = Annotation(
                                    image_id=image.id,
                                    dataset_id=dataset_id,
                                    class_name=class_name,
                                    bbox={
                                        "x": x,
                                        "y": y,
                                        "width": w,
                                        "height": h
                                    },
                                    x=x,
                                    y=y,
                                    width=w,
                                    height=h,
                                    area=w*h
                                )
                                
                                # Adicionar ao banco
                                # Adicionar ao banco (usar o objeto de sessão adequado aqui)
                                from microdetect.database.database import db_session
                                db_session.add(annotation)
                                count += 1
            
            # Commit das alterações
            from microdetect.database.database import db_session
            db_session.commit()
            
        elif import_format.lower() == "coco":
            # Implementar importação COCO
            pass
        
        return count 
from typing import List, Optional
from sqlalchemy.orm import Session
import os
import yaml
import shutil
from pathlib import Path

from microdetect.models.dataset import Dataset
from microdetect.models.annotation import Annotation
from microdetect.schemas.dataset import DatasetCreate, DatasetUpdate
from microdetect.core.config import settings


class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    def get(self, dataset_id: int) -> Optional[Dataset]:
        """
        Obter um dataset pelo ID.
        """
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def get_multi(self, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """
        Obter múltiplos datasets.
        """
        return self.db.query(Dataset).offset(skip).limit(limit).all()
    
    def create(self, dataset_in: DatasetCreate) -> Dataset:
        """
        Criar um novo dataset.
        """
        db_dataset = Dataset(**dataset_in.dict())
        self.db.add(db_dataset)
        self.db.commit()
        self.db.refresh(db_dataset)
        return db_dataset
    
    def update(self, db_dataset: Dataset, dataset_in: DatasetUpdate) -> Dataset:
        """
        Atualizar um dataset existente.
        """
        update_data = dataset_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_dataset, field, value)
        
        self.db.commit()
        self.db.refresh(db_dataset)
        return db_dataset
    
    def remove(self, dataset_id: int) -> None:
        """
        Remover um dataset.
        """
        db_dataset = self.get(dataset_id)
        if db_dataset:
            self.db.delete(db_dataset)
            self.db.commit()
    
    def prepare_for_training(self, dataset_id: int) -> str:
        """
        Prepara o dataset para treinamento com YOLO:
        - Cria a estrutura de diretórios
        - Copia as imagens para as pastas train, val, test
        - Converte as anotações para formato YOLO txt
        - Cria o arquivo data.yaml
        
        Args:
            dataset_id: ID do dataset
            
        Returns:
            Caminho para o arquivo data.yaml
        """
        dataset = self.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
        
        # Diretório base para o dataset
        dataset_dir = Path(f"data/datasets/{dataset_id}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar subdiretórios
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val" 
        test_dir = dataset_dir / "test"
        
        # Criar diretórios para imagens e labels
        train_images_dir = train_dir / "images"
        train_labels_dir = train_dir / "labels"
        val_images_dir = val_dir / "images"
        val_labels_dir = val_dir / "labels"
        test_images_dir = test_dir / "images"
        test_labels_dir = test_dir / "labels"
        
        # Garantir que todos os diretórios existam
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, 
                          test_images_dir, test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Obter todas as imagens e anotações do dataset
        images = dataset.associated_images
        
        # Dividir imagens em treino (70%), validação (20%) e teste (10%)
        total_images = len(images)
        train_count = int(total_images * 0.7)
        val_count = int(total_images * 0.2)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Processar imagens e anotações para cada conjunto
        self._process_images_and_annotations(train_images, train_images_dir, train_labels_dir, dataset_id)
        self._process_images_and_annotations(val_images, val_images_dir, val_labels_dir, dataset_id)
        self._process_images_and_annotations(test_images, test_images_dir, test_labels_dir, dataset_id)
        
        # Criar arquivo data.yaml
        data_yaml_path = dataset_dir / "data.yaml"
        data_yaml_content = {
            "path": str(dataset_dir.absolute()),
            "train": str(train_images_dir.absolute()),
            "val": str(val_images_dir.absolute()),
            "test": str(test_images_dir.absolute()),
            "nc": len(dataset.classes),
            "names": dataset.classes
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        return str(data_yaml_path)
    
    def _process_images_and_annotations(self, images, images_dir, labels_dir, dataset_id):
        """
        Processa as imagens e anotações para o formato YOLO.
        
        Args:
            images: Lista de imagens
            images_dir: Diretório para as imagens
            labels_dir: Diretório para as anotações (labels)
            dataset_id: ID do dataset
        """
        for image in images:
            # Copiar imagem para o diretório de destino
            src_image_path = Path(image.filepath)
            if not src_image_path.exists():
                continue
                
            dst_image_path = images_dir / src_image_path.name
            shutil.copy(src_image_path, dst_image_path)
            
            # Obter anotações da imagem e converter para formato YOLO
            annotations = self.db.query(Annotation).filter(
                Annotation.image_id == image.id,
                Annotation.dataset_id == dataset_id
            ).all()
            
            if not annotations:
                continue
                
            # Nome do arquivo de anotação (mesmo nome da imagem, mas com extensão .txt)
            label_filename = src_image_path.stem + ".txt"
            label_path = labels_dir / label_filename
            
            # Converter e salvar anotações no formato YOLO
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    # Formato YOLO: class_id x_center y_center width height
                    # Todos os valores são normalizados (0-1)
                    class_id = annotation.class_id
                    
                    # Extrair coordenadas do bbox (já normalizado ou precisa normalizar)
                    bbox = annotation.bbox  # Assumindo formato [x1, y1, x2, y2] normalizado
                    
                    # Converter para formato YOLO [x_center, y_center, width, height]
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    # Escrever no arquivo
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
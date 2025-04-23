import shutil
from pathlib import Path
from typing import List, Optional

import yaml
from sqlalchemy.orm import Session

from microdetect.core.config import settings
from microdetect.models.annotation import Annotation
from microdetect.models.dataset import Dataset
from microdetect.models.image import Image
from microdetect.models.dataset_image import DatasetImage
from microdetect.schemas.dataset import DatasetCreate, DatasetUpdate
import logging

logger = logging.getLogger(__name__)

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
        # Criar o diretório do dataset
        import os
        os.makedirs(dataset_in.path, exist_ok=True)
        
        # Criar o dataset no banco de dados
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
        dataset_dir = Path(settings.TRAINING_DIR) / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparando dataset {dataset.name} para treinamento em {dataset_dir}")
        
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
        
        # Obter todas as imagens do dataset usando a tabela intermediária dataset_images
        images = (
            self.db.query(Image)
            .join(DatasetImage, DatasetImage.image_id == Image.id)
            .filter(DatasetImage.dataset_id == dataset_id)
            .all()
        )
        
        if not images:
            raise ValueError(f"Dataset {dataset_id} não possui imagens")
            
        logger.info(f"Processando {len(images)} imagens do dataset {dataset.name}")
        
        # Dividir imagens em treino (70%), validação (20%) e teste (10%)
        total_images = len(images)
        train_count = int(total_images * 0.7)
        val_count = int(total_images * 0.2)
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        logger.info(f"Distribuição das imagens: {len(train_images)} treino, {len(val_images)} validação, {len(test_images)} teste")
        
        # Processar imagens e anotações para cada conjunto
        self._process_images_and_annotations(train_images, train_images_dir, train_labels_dir, dataset_id, dataset.classes)
        self._process_images_and_annotations(val_images, val_images_dir, val_labels_dir, dataset_id, dataset.classes)
        self._process_images_and_annotations(test_images, test_images_dir, test_labels_dir, dataset_id, dataset.classes)
        
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
        
        logger.info(f"Dataset preparado com sucesso. Arquivo data.yaml criado em {data_yaml_path}")
        
        return str(data_yaml_path)
    
    def _process_images_and_annotations(self, images, images_dir, labels_dir, dataset_id, classes):
        """
        Processa as imagens e anotações para o formato YOLO.
        
        Args:
            images: Lista de imagens
            images_dir: Diretório para as imagens
            labels_dir: Diretório para as anotações (labels)
            dataset_id: ID do dataset
            classes: Lista de nomes de classes para mapeamento
        """
        for image in images:
            # Copiar imagem para o diretório de destino
            src_image_path = Path(image.file_path)
            if not src_image_path.exists():
                logger.warning(f"Imagem não encontrada: {src_image_path}")
                continue
                
            dst_image_path = images_dir / src_image_path.name
            shutil.copy(src_image_path, dst_image_path)
            
            # Obter anotações da imagem e converter para formato YOLO
            annotations = self.db.query(Annotation).filter(
                Annotation.image_id == image.id,
                Annotation.dataset_id == dataset_id
            ).all()
            
            if not annotations:
                logger.warning(f"Imagem {image.id} não possui anotações")
                continue
                
            # Nome do arquivo de anotação (mesmo nome da imagem, mas com extensão .txt)
            label_filename = src_image_path.stem + ".txt"
            label_path = labels_dir / label_filename
            
            # Converter e salvar anotações no formato YOLO
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    try:
                        # Formato YOLO: class_id x_center y_center width height
                        # Todos os valores são normalizados (0-1)
                        class_id = classes.index(annotation.class_name)
                        
                        # Verificar se a imagem tem dimensões
                        if not image.width or not image.height:
                            logger.warning(f"Imagem {image.id} não tem dimensões definidas")
                            continue

                        # Converter coordenadas para o formato YOLO (normalizadas)
                        x_center = (annotation.x + (annotation.width / 2)) / image.width
                        y_center = (annotation.y + (annotation.height / 2)) / image.height
                        width = annotation.width / image.width
                        height = annotation.height / image.height

                        # Verificar se os valores estão dentro do intervalo [0,1]
                        if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                            logger.warning(f"Anotação {annotation.id} tem coordenadas fora do intervalo [0,1]")
                            continue

                        # Escrever no arquivo
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    except Exception as e:
                        logger.error(f"Erro ao processar anotação {annotation.id}: {str(e)}")
                        continue
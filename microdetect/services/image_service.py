import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from microdetect.core.config import settings
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset
from microdetect.models.dataset_image import DatasetImage
import cv2
import numpy as np
import logging
from PIL import Image as PILImage
import io

logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        self.images_dir = Path(settings.IMAGES_DIR)
        self.gallery_dir = Path(settings.GALLERY_DIR)
        self.temp_dir = Path(settings.TEMP_DIR)
        
        # Criar diretórios se não existirem
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image_data,
        filename: str,
        dataset_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Salva uma imagem no sistema de arquivos.
        
        Args:
            image_data: Dados binários da imagem
            filename: Nome do arquivo
            dataset_id: ID do dataset (opcional)
            metadata: Metadados adicionais da imagem (opcional)
            width: Largura da imagem (opcional)
            height: Altura da imagem (opcional)
            
        Returns:
            Dicionário com informações da imagem salva
        """
        # Determinar diretório de destino
        dest_dir = self.images_dir
        if dataset_id:
            dest_dir = dest_dir / 'datasets' / str(dataset_id)
            
        # Criar diretório se não existir
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Gerar nome de arquivo único se necessário
        if not filename:
            filename = f"{uuid.uuid4()}.jpg"
        
        # Garantir que o nome do arquivo seja único
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        counter = 0
        final_filename = filename
        
        # Verificar se o arquivo já existe e gerar nome único
        while (dest_dir / final_filename).exists():
            counter += 1
            final_filename = f"{base_name}_{counter}{extension}"
            
        # Se as dimensões não foram fornecidas, extraí-las da imagem
        img_dimensions = None
        if width is None or height is None:
            try:
                # Abrir a imagem usando PIL para obter dimensões
                img = PILImage.open(io.BytesIO(image_data))
                img_dimensions = img.size
                # Fechar a imagem para liberar recursos
                img.close()
            except Exception as e:
                print(f"Erro ao extrair dimensões da imagem: {e}")
        
        # Usar dimensões extraídas se não fornecidas
        if width is None and img_dimensions:
            width = img_dimensions[0]
        
        if height is None and img_dimensions:
            height = img_dimensions[1]
            
        # Salvar arquivo
        filepath = dest_dir / final_filename
        with open(filepath, "wb") as f:
            f.write(image_data)
            
        # Obter tamanho do arquivo
        filesize = filepath.stat().st_size
        
        # Preparar metadados
        image_metadata = {
            "timestamp": datetime.now().isoformat(),
            "file_size": filesize,
            "format": extension.lstrip('.').lower(),
        }
        
        # Adicionar dimensões aos metadados se disponíveis
        if width is not None:
            image_metadata["width"] = width
        
        if height is not None:
            image_metadata["height"] = height
        
        # Mesclar com metadados fornecidos
        if metadata:
            image_metadata.update(metadata)
        
        # URL da imagem para acesso via API
        url_path = f"http://localhost:8000/images"

        if dataset_id:
            url_path = f"{url_path}/datasets/{dataset_id}"

        url_path = f"{url_path}/{final_filename}"
        
        # Criar info básica do arquivo
        image_info = {
            "file_name": final_filename,
            "file_path": str(filepath),
            "file_size": filesize,
            "dataset_id": dataset_id,
            "url": url_path,
            "image_metadata": image_metadata
        }
        
        # Adicionar dimensões diretamente ao image_info se disponíveis
        if width is not None:
            image_info["width"] = width
        
        if height is not None:
            image_info["height"] = height
        
        return image_info

    async def get_image(self, image_id: int) -> Image:
        """
        Recupera uma imagem do banco de dados.
        
        Args:
            image_id: ID da imagem
            
        Returns:
            Objeto Image
        """
        image = Image.query.get(image_id)
        if not image:
            raise ValueError(f"Imagem {image_id} não encontrada")
        return image

    def delete_image(self, filename, dataset_id=None) -> bool:
        """
        Remove uma imagem do sistema de arquivos.
        
        Args:
            filename: Nome do arquivo
            dataset_id: ID do dataset (opcional)
            
        Returns:
            True se a imagem foi removida, False caso contrário
        """
        # Determinar caminho do arquivo
        dest_dir = self.images_dir
        if dataset_id:
            dest_dir = dest_dir / str(dataset_id)
            
        filepath = dest_dir / filename
        
        # Verificar se o arquivo existe
        if not filepath.exists():
            return False
        
        # Remover arquivo
        try:
            filepath.unlink()
            return True
        except Exception:
            return False

    async def list_images(
        self,
        dataset_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Image]:
        """
        Lista imagens do banco de dados.
        
        Args:
            dataset_id: ID do dataset (opcional)
            skip: Número de registros para pular
            limit: Número máximo de registros
            
        Returns:
            Lista de objetos Image
        """
        query = Image.query
        
        if dataset_id:
            # Usar a tabela DatasetImage para filtrar por dataset_id
            query = query.join(DatasetImage, Image.id == DatasetImage.image_id).filter(
                DatasetImage.dataset_id == dataset_id
            )
        
        return query.offset(skip).limit(limit).all()

    async def move_image(
        self,
        image_id: int,
        new_dataset_id: int
    ) -> Image:
        """
        Move uma imagem para outro dataset.
        
        Args:
            image_id: ID da imagem
            new_dataset_id: ID do novo dataset
            
        Returns:
            Objeto Image atualizado
        """
        # Verificar imagem
        image = await self.get_image(image_id)
        
        # Verificar novo dataset
        new_dataset = Dataset.query.get(new_dataset_id)
        if not new_dataset:
            raise ValueError(f"Dataset {new_dataset_id} não encontrado")
        
        # Criar diretório do novo dataset se não existir
        new_dataset_dir = self.images_dir / str(new_dataset_id)
        new_dataset_dir.mkdir(exist_ok=True)
        
        # Mover arquivo
        new_filepath = new_dataset_dir / image.file_name
        shutil.move(image.file_path, new_filepath)
        
        # Atualizar registro
        # Não atualizar dataset_id diretamente, pois não existe no modelo Image
        image.file_path = str(new_filepath)
        
        return image

    async def get_image_info(self, image_id: int) -> Dict[str, Any]:
        """
        Obtém informações sobre uma imagem.
        
        Args:
            image_id: ID da imagem
            
        Returns:
            Dicionário com informações da imagem
        """
        image = await self.get_image(image_id)
        
        return {
            "id": image.id,
            "file_name": image.file_name,
            "file_path": image.file_path,
            "file_size": image.file_size,
            "url": image.url,
            "dataset_id": image.dataset_id,
            "created_at": image.created_at,
            "updated_at": image.updated_at,
            "metadata": image.metadata,
            "annotations_count": len(image.annotations) if hasattr(image, 'annotations') else 0,
            "inference_results_count": len(image.inference_results) if hasattr(image, 'inference_results') else 0
        }

    async def copy_or_move_image_to_dataset(
        self,
        image_id: int,
        dataset_id: int,
        db
    ) -> Dict[str, Any]:
        """
        Associa uma imagem a um dataset, movendo-a ou copiando-a conforme necessário.
        
        - Se a imagem não estiver associada a nenhum dataset, move o arquivo.
        - Se a imagem já estiver associada a outro dataset, copia o arquivo.
        
        Args:
            image_id: ID da imagem
            dataset_id: ID do dataset
            db: Sessão do banco de dados
            
        Returns:
            Dicionário com informações da associação
        """
        # Verificar se a imagem existe
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise ValueError(f"Imagem {image_id} não encontrada")
            
        # Verificar se o dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} não encontrado")
            
        # Verificar se já existe uma associação
        existing_association = (
            db.query(DatasetImage)
            .filter(
                DatasetImage.image_id == image_id,
                DatasetImage.dataset_id == dataset_id
            )
            .first()
        )
        
        if existing_association:
            return {
                "id": existing_association.id,
                "dataset_id": dataset_id,
                "image_id": image_id,
                "created_at": existing_association.created_at,
                "message": "Associação já existe"
            }
            
        # Determinar diretório de destino
        dest_dir = self.images_dir / 'datasets' / str(dataset_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Fonte: caminho atual da imagem
        source_path = Path(image.file_path)
        
        # Verificar se o arquivo existe
        if not source_path.exists():
            raise ValueError(f"Arquivo de imagem não encontrado: {source_path}")
            
        # Nome do arquivo no destino (usar o nome original)
        dest_filename = source_path.name
        
        # Caminho completo de destino
        dest_path = dest_dir / dest_filename
        
        # Garantir que o nome do arquivo seja único no destino
        counter = 0
        final_dest_path = dest_path

        final_file_name = dest_filename
        while final_dest_path.exists():
            counter += 1
            name_parts = dest_filename.rsplit('.', 1)
            if len(name_parts) > 1:
                base, ext = name_parts
                final_file_name = f"{base}_{counter}.{ext}"
                final_dest_path = dest_dir / final_file_name
            else:
                final_file_name = f"{dest_filename}_{counter}"
                final_dest_path = dest_dir / final_file_name

        final_url = f"http://localhost:8000/images/datasets/{str(dataset_id)}/{final_file_name}"
        # Decidir se deve copiar ou mover o arquivo
        if image.dataset_id is None:
            # Se a imagem não estiver associada a nenhum dataset, mover
            shutil.move(str(source_path), str(final_dest_path))
            
            # Atualizar o caminho do arquivo
            image.file_path = str(final_dest_path)

            image.url = final_url
            
            # Atualizar o dataset_id (para compatibilidade)
            image.dataset_id = dataset_id
        else:
            # Se a imagem já estiver associada a outro dataset, copiar
            shutil.copy2(str(source_path), str(final_dest_path))
            
        # Criar a associação na tabela de relacionamento
        association = DatasetImage(
            dataset_id=dataset_id,
            image_id=image_id,
            path=str(final_dest_path)
        )
        
        db.add(association)
        db.commit()
        db.refresh(association)
        
        return {
            "id": association.id,
            "dataset_id": dataset_id,
            "image_id": image_id,
            "file_path": str(final_dest_path),
            "url": final_url,
            "created_at": association.created_at,
            "message": "Imagem associada com sucesso"
        } 
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import List, Dict, Any
from datetime import datetime
from microdetect.database.database import get_db
from microdetect.models.dataset import Dataset
from microdetect.models.image import Image
from microdetect.models.dataset_image import DatasetImage
from microdetect.models.annotation import Annotation
from microdetect.schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpdate
from microdetect.schemas.dataset_image import DatasetImageResponse
from microdetect.schemas.dataset_statistics import DatasetStatistics
from microdetect.services.image_service import ImageService
from microdetect.services.dataset_service import DatasetService
from microdetect.schemas.class_distribution import ClassDistributionResponse, ClassInfo
import json

router = APIRouter()
image_service = ImageService()

@router.post("/", response_model=DatasetResponse)
def create_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_in: DatasetCreate,
) -> Any:
    """
    Criar um novo dataset.
    """
    dataset = DatasetService(db).create(dataset_in)
    return dataset

@router.get("/", response_model=List[DatasetResponse])
def list_datasets(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Recuperar todos os datasets.
    """
    datasets = DatasetService(db).get_multi(skip=skip, limit=limit)
    
    # Para cada dataset, carregar a contagem de imagens
    for dataset in datasets:
        dataset.images_count = db.query(func.count(Image.id)).filter(Image.dataset_id == dataset.id).scalar() or 0
    
    return datasets

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
) -> Any:
    """
    Recuperar um dataset específico pelo ID.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Carregar a contagem de imagens
    dataset.images_count = db.query(func.count(Image.id)).filter(Image.dataset_id == dataset.id).scalar() or 0
    
    return dataset

@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    dataset_in: DatasetUpdate,
) -> Any:
    """
    Atualizar um dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    dataset = DatasetService(db).update(dataset, dataset_in)
    
    # Carregar a contagem de imagens
    dataset.images_count = db.query(func.count(Image.id)).filter(Image.dataset_id == dataset.id).scalar() or 0
    
    return dataset

@router.delete("/{dataset_id}")
def delete_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
) -> Any:
    """
    Excluir um dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    DatasetService(db).remove(dataset_id)
    
    return {"success": True}

@router.get("/{dataset_id}/stats", response_model=DatasetStatistics)
def get_dataset_statistics(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
) -> Any:
    """
    Obter estatísticas para um dataset específico.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Contagem de imagens
    total_images = db.query(func.count(Image.id)).filter(Image.dataset_id == dataset_id).scalar() or 0
    
    # Contagem total de anotações
    total_annotations = db.query(func.count(Annotation.id)) \
        .join(Image, Annotation.image_id == Image.id) \
        .filter(Image.dataset_id == dataset_id) \
        .scalar() or 0
    
    # Imagens com anotações vs. sem anotações
    annotated_images_query = db.query(func.count(func.distinct(Image.id))) \
        .join(Annotation, Annotation.image_id == Image.id) \
        .filter(Image.dataset_id == dataset_id)
    
    annotated_images = annotated_images_query.scalar() or 0
    unannotated_images = total_images - annotated_images
    
    # Distribuição de classes (para class_counts)
    class_counts = {}
    
    if total_annotations > 0:
        # Buscar todas as classes usadas nas anotações do dataset
        class_counts_result = db.execute(
            text("""
            SELECT a.class_name, COUNT(*) as count
            FROM annotation a
            JOIN image i ON a.image_id = i.id
            WHERE i.dataset_id = :dataset_id
            GROUP BY a.class_name
            ORDER BY count DESC
            """),
            {"dataset_id": dataset_id}
        ).fetchall()
        
        # Calcular distribuição
        for row in class_counts_result:
            class_name = row[0]
            count = row[1]
            class_counts[class_name] = count
    
    # Média de objetos por imagem
    average_objects_per_image = None
    if total_images > 0:
        average_objects_per_image = total_annotations / total_images
    
    # Tamanho médio das imagens (se disponível)
    average_image_size = None
    try:
        size_query_result = db.execute(
            text("""
            SELECT AVG(i.width) as avg_width, AVG(i.height) as avg_height
            FROM image i
            WHERE i.dataset_id = :dataset_id AND i.width IS NOT NULL AND i.height IS NOT NULL
            """),
            {"dataset_id": dataset_id}
        ).fetchone()
        
        if size_query_result and size_query_result[0] and size_query_result[1]:
            average_image_size = {
                "width": round(size_query_result[0]),
                "height": round(size_query_result[1])
            }
    except Exception:
        # Se houver erro ou se width/height não existirem na tabela, ignorar
        pass
    
    # Distribuição de tamanhos de objetos
    object_size_distribution = None
    
    # Tenta extrair informações de tamanho dos bounding boxes do campo JSON 'bbox'
    try:
        # Buscar todos os bounding boxes e informações de imagem
        bbox_data = db.execute(
            text("""
            SELECT a.bbox, i.width, i.height
            FROM annotation a
            JOIN image i ON a.image_id = i.id
            WHERE i.dataset_id = :dataset_id AND a.bbox IS NOT NULL
            """),
            {"dataset_id": dataset_id}
        ).fetchall()
        
        if bbox_data:
            small_objects = 0
            medium_objects = 0
            large_objects = 0
            
            total_obj_area = 0
            total_img_area = 0
            
            for row in bbox_data:
                bbox_json = row[0]
                img_width = row[1]
                img_height = row[2]
                
                if not bbox_json or not img_width or not img_height:
                    continue
                
                try:
                    # Tentar extrair as dimensões do bbox (pode estar em diferentes formatos)
                    bbox = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
                    
                    # Verifica formato [x_min, y_min, width, height] ou [x_min, y_min, x_max, y_max]
                    if len(bbox) == 4:
                        if isinstance(bbox, list):
                            # Formato [x_min, y_min, x_max, y_max]
                            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                                bbox_width = bbox[2] - bbox[0]
                                bbox_height = bbox[3] - bbox[1]
                            # Formato [x_min, y_min, width, height]
                            else:
                                bbox_width = bbox[2]
                                bbox_height = bbox[3]
                        elif isinstance(bbox, dict):
                            # Formato com keys específicas
                            if 'width' in bbox and 'height' in bbox:
                                bbox_width = bbox['width']
                                bbox_height = bbox['height']
                            elif 'x_min' in bbox and 'x_max' in bbox:
                                bbox_width = bbox['x_max'] - bbox['x_min']
                                bbox_height = bbox['y_max'] - bbox['y_min']
                            else:
                                continue
                        else:
                            continue
                        
                        # Calcular áreas
                        obj_area = bbox_width * bbox_height
                        img_area = img_width * img_height
                        
                        total_obj_area += obj_area
                        total_img_area += img_area
                        
                        # Classificar objetos por tamanho relativo
                        area_ratio = obj_area / img_area
                        
                        if area_ratio < 0.1:
                            small_objects += 1
                        elif area_ratio < 0.3:
                            medium_objects += 1
                        else:
                            large_objects += 1
                            
                except (json.JSONDecodeError, TypeError, IndexError, KeyError):
                    # Ignora bboxes em formato desconhecido
                    continue
            
            # Criar distribuição de tamanhos se houver objetos classificados
            total_sized_objects = small_objects + medium_objects + large_objects
            if total_sized_objects > 0:
                object_size_distribution = {
                    "small": small_objects,
                    "medium": medium_objects,
                    "large": large_objects
                }
                
                # Calcular densidade média de objetos
                average_object_density = None
                if total_img_area > 0:
                    average_object_density = total_obj_area / total_img_area
    except Exception:
        # Ignorar erros ao processar bboxes
        pass
    
    # Calcular desbalanceamento de classes
    class_imbalance = None
    if class_counts and len(class_counts) > 1:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        if max_count > 0:
            class_imbalance = 1.0 - (min_count / max_count)
    
    # Timestamp atual para last_calculated
    last_calculated = datetime.utcnow()
    
    # Construir e retornar as estatísticas
    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "annotated_images": annotated_images,
        "unannotated_images": unannotated_images,
        "average_image_size": average_image_size,
        "object_size_distribution": object_size_distribution,
        "class_imbalance": class_imbalance,
        "average_objects_per_image": average_objects_per_image,
        "average_object_density": average_object_density if 'average_object_density' in locals() else None,
        "last_calculated": last_calculated,
        "class_counts": class_counts,
        "extra_data": None
    }

@router.post("/{dataset_id}/classes")
def add_class(
        *,
        db: Session = Depends(get_db),
        dataset_id: int,
        class_data: Dict[str, str] = Body(...),
) -> Any:
    """
    Adicionar uma classe ao dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")

    class_name = class_data.get("class_name")
    if not class_name:
        raise HTTPException(status_code=400, detail="Nome da classe é obrigatório")

    # Verificar se a classe já existe no dataset
    if dataset.classes and class_name in dataset.classes:
        raise HTTPException(status_code=400, detail=f"Classe '{class_name}' já existe no dataset")

    # Adicionar a classe ao dataset
    if not dataset.classes:
        dataset.classes = [class_name]
    else:
        # Criar uma nova lista para garantir que a mudança seja detectada
        current_classes = list(dataset.classes)
        current_classes.append(class_name)
        dataset.classes = current_classes

    # Marcar objeto como modificado
    from sqlalchemy import inspect
    inspect(dataset).modified = True  # Força o objeto a ser considerado modificado

    db.commit()
    db.refresh(dataset)

    return {"success": True, "message": f"Classe '{class_name}' adicionada com sucesso"}

@router.delete("/{dataset_id}/classes/{class_name}")
def remove_class(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    class_name: str,
) -> Any:
    """
    Remover uma classe do dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Verificar se a classe existe no dataset
    if not dataset.classes or class_name not in dataset.classes:
        raise HTTPException(status_code=404, detail=f"Classe '{class_name}' não encontrada no dataset")
    
    # Remover a classe do dataset
    dataset.classes.remove(class_name)
    db.commit()
    db.refresh(dataset)
    
    return {"success": True, "message": f"Classe '{class_name}' removida com sucesso"}

@router.get("/{dataset_id}/class-distribution", response_model=List[ClassDistributionResponse])
def get_class_distribution(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
) -> Any:
    """
    Obter a distribuição de classes para um dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    result = []
    
    # Verificar se o dataset tem classes definidas
    defined_classes = dataset.classes if dataset.classes else []
    
    # Criar dicionário para armazenar contagens
    class_counts_dict = {}
    
    # Verificar se a tabela annotations existe
    try:
        table_exists = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='annotations'")
        ).fetchone() is not None
        
        if table_exists:
            # Tentar obter todas as classes usadas nas anotações
            class_counts = db.execute(
                text("""
                SELECT a.class_name, COUNT(*) as count
                FROM annotations a
                JOIN images i ON a.image_id = i.id
                WHERE i.dataset_id = :dataset_id
                GROUP BY a.class_name
                """),
                {"dataset_id": dataset_id}
            ).fetchall()
            
            # Preencher o dicionário com os resultados
            class_counts_dict = {row[0]: row[1] for row in class_counts}
        else:
            import logging
            logging.info("Tabela de anotações ainda não existe. Retornando apenas classes definidas sem contagens.")
    except Exception as e:
        # Se ocorrer erro, continuar sem contagens
        import logging
        logging.warning(f"Erro ao buscar distribuição de classes: {str(e)}")
        # Continuar com um dicionário vazio
    
    # Total de anotações
    total_annotations = sum(class_counts_dict.values()) if class_counts_dict else 0
    
    # Adicionar classes definidas, mesmo que não tenham anotações
    for class_name in defined_classes:
        count = class_counts_dict.get(class_name, 0)
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        
        result.append(
            ClassInfo(
                class_name=class_name,
                count=count,
                percentage=percentage,
                is_used=count > 0,
                is_undefined=False
            )
        )
    
    # Adicionar classes usadas nas anotações, mas não definidas no dataset
    for class_name, count in class_counts_dict.items():
        if class_name not in defined_classes:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            
            result.append(
                ClassInfo(
                    class_name=class_name,
                    count=count,
                    percentage=percentage,
                    is_used=True,
                    is_undefined=True
                )
            )
    
    # Ordenar por contagem (decrescente)
    result.sort(key=lambda x: x.count, reverse=True)
    
    return result

@router.post("/{dataset_id}/images", response_model=DatasetImageResponse)
async def associate_image_to_dataset(
    dataset_id: int,
    image_id: int = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Associa uma imagem existente a um dataset."""
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Verificar se a imagem já está associada ao dataset
    existing_association = db.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id,
        DatasetImage.image_id == image_id
    ).first()
    
    if existing_association:
        # Se já existe, apenas retornar a associação existente
        return existing_association

    # Fazer a associação usando o serviço de imagens
    try:
        image_service = ImageService()
        result = await image_service.copy_or_move_image_to_dataset(
            image_id=image_id,
            dataset_id=dataset_id,
            db=db
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao associar imagem: {str(e)}")

@router.delete("/{dataset_id}/images/{image_id}")
async def remove_image_from_dataset(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db)
):
    """Remove uma imagem de um dataset."""
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Verificar se a imagem está associada ao dataset
    association = db.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id,
        DatasetImage.image_id == image_id
    ).first()
    
    if not association:
        raise HTTPException(
            status_code=404,
            detail="Imagem não está associada a este dataset"
        )
    
    # Remover a associação e o arquivo físico se necessário
    try:
        # Primeiro, verificar se a imagem está associada apenas a este dataset
        other_associations_count = db.query(DatasetImage).filter(
            DatasetImage.image_id == image_id,
            DatasetImage.dataset_id != dataset_id
        ).count()
        
        # Remover a associação do banco de dados
        db.delete(association)
        db.commit()
        
        # Se não houver outras associações, remover o arquivo físico
        if other_associations_count == 0 and association.file_path:
            import os
            if os.path.exists(association.file_path):
                os.remove(association.file_path)
        
        return {"message": "Imagem removida do dataset com sucesso"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao remover imagem do dataset: {str(e)}"
        ) 
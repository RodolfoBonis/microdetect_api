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
from microdetect.utils.serializers import build_response, build_error_response
import json

router = APIRouter()
image_service = ImageService()

@router.post("/", response_model=None)
def create_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_in: dict,
) -> Any:
    """
    Criar um novo dataset.
    """
    # Criar instância de DatasetCreate a partir do dict recebido
    dataset_create = DatasetCreate(**dataset_in)
    
    # Criar dataset
    dataset = DatasetService(db).create(dataset_create)
    
    # Converter para esquema de resposta
    response = DatasetResponse.from_orm(dataset)
    return build_response(response)

@router.get("/", response_model=None)
def list_datasets(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Recuperar todos os datasets.
    """
    datasets = DatasetService(db).get_multi(skip=skip, limit=limit)
    
    # Para cada dataset, carregar a contagem de imagens e anotações
    for dataset in datasets:
        # Contagem de imagens
        dataset.images_count = db.query(func.count(DatasetImage.id)).filter(DatasetImage.dataset_id == dataset.id).scalar() or 0
        
        # Contagem de anotações
        dataset.annotations_count = db.query(func.count(Annotation.id)) \
            .join(Image, Annotation.image_id == Image.id) \
            .join(DatasetImage, DatasetImage.image_id == Image.id) \
            .filter(DatasetImage.dataset_id == dataset.id) \
            .scalar() or 0
        
        # Buscar a primeira imagem do dataset para usar como thumb
        first_image = db.query(Image) \
            .join(DatasetImage, DatasetImage.image_id == Image.id) \
            .filter(DatasetImage.dataset_id == dataset.id) \
            .order_by(Image.id) \
            .first()
        
        # Adicionar atributo thumb com a URL da imagem ou string vazia
        dataset.thumb = first_image.url if first_image else ""
    
    # Converter para esquema de resposta
    response_list = [DatasetResponse.from_orm(ds) for ds in datasets]
    return build_response(response_list)

@router.get("/{dataset_id}", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
    # Carregar a contagem de imagens
    dataset.images_count = db.query(func.count(DatasetImage.id)).filter(DatasetImage.dataset_id == dataset.id).scalar() or 0
    
    # Carregar a contagem de anotações
    dataset.annotations_count = db.query(func.count(Annotation.id)) \
        .join(Image, Annotation.image_id == Image.id) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset.id) \
        .scalar() or 0
    
    # Buscar a primeira imagem do dataset para usar como thumb
    first_image = db.query(Image) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset.id) \
        .order_by(Image.id) \
        .first()
    
    # Adicionar atributo thumb com a URL da imagem ou string vazia
    dataset.thumb = first_image.url if first_image else ""
    
    # Converter para esquema de resposta
    response = DatasetResponse.from_orm(dataset)
    return build_response(response)

@router.put("/{dataset_id}", response_model=None)
def update_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    dataset_in: dict,
) -> Any:
    """
    Atualizar um dataset.
    """
    dataset = DatasetService(db).get(dataset_id)
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    # Criar instância de DatasetUpdate a partir do dict recebido
    dataset_update = DatasetUpdate(**dataset_in)
    
    # Atualizar dataset
    dataset = DatasetService(db).update(dataset, dataset_update)
    
    # Carregar a contagem de imagens
    dataset.images_count = db.query(func.count(DatasetImage.id)).filter(DatasetImage.dataset_id == dataset.id).scalar() or 0
    
    # Carregar a contagem de anotações
    dataset.annotations_count = db.query(func.count(Annotation.id)) \
        .join(Image, Annotation.image_id == Image.id) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset.id) \
        .scalar() or 0
    
    # Buscar a primeira imagem do dataset para usar como thumb
    first_image = db.query(Image) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset.id) \
        .order_by(Image.id) \
        .first()
    
    # Adicionar atributo thumb com a URL da imagem ou string vazia
    dataset.thumb = first_image.url if first_image else ""
    
    # Converter para esquema de resposta
    response = DatasetResponse.from_orm(dataset)
    return build_response(response)

@router.delete("/{dataset_id}", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
    DatasetService(db).remove(dataset_id)
    
    return build_response({"success": True})

@router.get("/{dataset_id}/stats", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
    # Contagem de imagens
    total_images = db.query(func.count(DatasetImage.id)).filter(DatasetImage.dataset_id == dataset_id).scalar() or 0
    
    # Contagem total de anotações
    total_annotations = db.query(func.count(Annotation.id)) \
        .join(Image, Annotation.image_id == Image.id) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset_id) \
        .scalar() or 0
    
    # Imagens com anotações vs. sem anotações
    annotated_images_query = db.query(func.count(func.distinct(Image.id))) \
        .join(Annotation, Annotation.image_id == Image.id) \
        .join(DatasetImage, DatasetImage.image_id == Image.id) \
        .filter(DatasetImage.dataset_id == dataset_id)
    
    annotated_images = annotated_images_query.scalar() or 0
    unannotated_images = total_images - annotated_images
    
    # Distribuição de classes (para class_counts)
    class_counts = {}
    
    # Adicionar classes definidas no dataset, mesmo sem anotações
    if dataset.classes:
        for class_name in dataset.classes:
            class_counts[class_name] = 0  # Inicializa com zero anotações
    
    if total_annotations > 0:
        # Buscar todas as classes usadas nas anotações do dataset
        class_counts_result = db.execute(
            text("""
            SELECT a.class_name, COUNT(*) as count
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            JOIN dataset_images di ON di.image_id = i.id
            WHERE di.dataset_id = :dataset_id
            GROUP BY a.class_name
            ORDER BY count DESC
            """),
            {"dataset_id": dataset_id}
        ).fetchall()
        
        # Atualizar contagens (só atualiza se já existirem anotações)
        for row in class_counts_result:
            class_name = row[0]
            count = row[1]
            class_counts[class_name] = count
    
    # Média de objetos por imagem
    average_objects_per_image = None
    if total_images > 0:
        average_objects_per_image = total_annotations / total_images
    
    # Tamanho médio das imagens (se disponível)
    average_image_size = {"width": 0, "height": 0}  # Valor padrão
    
    try:
        # Primeiro, vamos verificar se as colunas width e height existem na tabela image
        import logging
        
        # Listar todas as imagens do dataset para diagnóstico
        images_data = db.execute(
            text("""
            SELECT i.id, i.file_name, i.width, i.height 
            FROM images i
            JOIN dataset_images di ON di.image_id = i.id
            WHERE di.dataset_id = :dataset_id
            LIMIT 10
            """),
            {"dataset_id": dataset_id}
        ).fetchall()
        
        logging.info(f"Dados de imagens para dataset {dataset_id}: {images_data}")
        
        # Se temos pelo menos uma imagem
        if images_data:
            # Abordagem 1: Tentar calcular média usando SQL
            size_query_result = db.execute(
                text("""
                SELECT AVG(CAST(i.width AS FLOAT)) as avg_width, AVG(CAST(i.height AS FLOAT)) as avg_height
                FROM images i
                JOIN dataset_images di ON di.image_id = i.id
                WHERE di.dataset_id = :dataset_id
                """),
                {"dataset_id": dataset_id}
            ).fetchone()
            
            logging.info(f"Resultado da query de tamanho médio: {size_query_result}")
            
            if size_query_result and size_query_result[0] and size_query_result[1]:
                average_image_size = {
                    "width": round(size_query_result[0]),
                    "height": round(size_query_result[1])
                }
                logging.info(f"Tamanho médio calculado pelo SQL: {average_image_size}")
            else:
                # Abordagem 2: Calcular manualmente a partir dos dados individuais
                valid_images = [img for img in images_data if img[2] and img[3] and img[2] > 0 and img[3] > 0]
                
                if valid_images:
                    total_width = sum(img[2] for img in valid_images)
                    total_height = sum(img[3] for img in valid_images)
                    count = len(valid_images)
                    
                    average_image_size = {
                        "width": round(total_width / count),
                        "height": round(total_height / count)
                    }
                    logging.info(f"Tamanho médio calculado manualmente: {average_image_size}")
                else:
                    logging.warning(f"Nenhuma imagem com dimensões válidas encontrada para o dataset {dataset_id}")
    except Exception as e:
        import logging
        logging.error(f"Erro ao calcular tamanho médio de imagens: {str(e)}")
        # Manter valor padrão
    
    # Distribuição de tamanhos de objetos
    object_size_distribution = None
    
    # Inicializar a densidade média de objetos com valor padrão antes do bloco try
    average_object_density = 0
    
    # Tenta extrair informações de tamanho dos bounding boxes do campo JSON 'bbox'
    try:
        # Buscar todos os bounding boxes e informações de imagem
        bbox_data = db.execute(
            text("""
            SELECT a.x, a.y, a.width, a.height, i.width as img_width, i.height as img_height
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            JOIN dataset_images di ON di.image_id = i.id
            WHERE di.dataset_id = :dataset_id 
            AND a.x IS NOT NULL AND a.y IS NOT NULL
            AND a.width IS NOT NULL AND a.height IS NOT NULL
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
                # Extrair valores diretamente das colunas
                x = row[0]
                y = row[1]
                bbox_width = row[2]
                bbox_height = row[3]
                img_width = row[4]
                img_height = row[5]
                
                if not x or not y or not bbox_width or not bbox_height or not img_width or not img_height:
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
            
            # Criar distribuição de tamanhos se houver objetos classificados
            total_sized_objects = small_objects + medium_objects + large_objects
            if total_sized_objects > 0:
                object_size_distribution = {
                    "small": small_objects,
                    "medium": medium_objects,
                    "large": large_objects
                }
                
                # Calcular densidade média de objetos
                if total_img_area > 0:
                    average_object_density = total_obj_area / total_img_area
    except Exception as e:
        # Log do erro para facilitar depuração
        import logging
        logging.error(f"Erro ao processar tamanhos de objetos: {str(e)}")
        # Mantém o valor padrão já inicializado de average_object_density
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
    
    # Criar instância de DatasetStatistics com as estatísticas calculadas
    stats = DatasetStatistics(
        total_images=total_images,
        total_annotations=total_annotations,
        annotated_images=annotated_images,
        unannotated_images=unannotated_images,
        average_image_size=average_image_size,
        object_size_distribution=object_size_distribution,
        class_imbalance=class_imbalance,
        average_objects_per_image=average_objects_per_image,
        average_object_density=average_object_density,
        last_calculated=last_calculated,
        class_counts=class_counts,
    )
    
    return build_response(stats)

@router.post("/{dataset_id}/classes", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
    # Extrair nome da classe do corpo da requisição
    class_name = class_data.get("class_name")
    if not class_name:
        return build_error_response("Nome da classe não fornecido", 400)
    
    # Verificar se a classe já existe no dataset
    if dataset.classes and class_name in dataset.classes:
        return build_error_response(f"Classe '{class_name}' já existe no dataset", 400)

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

    return build_response({"success": True, "message": f"Classe '{class_name}' adicionada com sucesso"})

@router.delete("/{dataset_id}/classes/{class_name}", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
    # Verificar se a classe existe no dataset
    if not dataset.classes or class_name not in dataset.classes:
        return build_error_response(f"Classe '{class_name}' não encontrada no dataset", 404)
    
    # Remover a classe do dataset
    dataset.classes.remove(class_name)
    db.commit()
    db.refresh(dataset)
    
    return build_response({"success": True, "message": f"Classe '{class_name}' removida com sucesso"})

@router.get("/{dataset_id}/class-distribution", response_model=None)
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
        return build_error_response("Dataset não encontrado", 404)
    
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
                JOIN dataset_images di ON di.image_id = i.id
                WHERE di.dataset_id = :dataset_id
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
    
    return build_response(result)

@router.post("/{dataset_id}/images", response_model=None)
async def associate_image_to_dataset(
    dataset_id: int,
    image_id: int = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Associa uma imagem existente a um dataset."""
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        return build_error_response("Imagem não encontrada", 404)
    
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
        
        response = DatasetImageResponse.from_orm(result)
        return build_response(response)
    except Exception as e:
        return build_error_response(f"Erro ao associar imagem: {str(e)}", 500)

@router.delete("/{dataset_id}/images/{image_id}", response_model=None)
async def remove_image_from_dataset(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db)
):
    """Remove uma imagem de um dataset."""
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        return build_error_response("Imagem não encontrada", 404)
    
    # Verificar se a imagem está associada ao dataset
    association = db.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id,
        DatasetImage.image_id == image_id
    ).first()
    
    if not association:
        return build_error_response("Imagem não está associada ao dataset", 404)
    
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
        
        return build_response({"success": True, "message": "Imagem removida do dataset com sucesso"})
    except Exception as e:
        db.rollback()
        return build_error_response(f"Erro ao remover imagem do dataset: {str(e)}", 500) 
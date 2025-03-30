from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
from microdetect.database.database import get_db
from microdetect.models.annotation import Annotation
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset
from microdetect.schemas.annotation import (
    AnnotationCreate, AnnotationUpdate, 
    AnnotationResponse, AnnotationBatch
)
from microdetect.utils.serializers import build_response, build_error_response

router = APIRouter()

@router.post("/", response_model=None)
def create_annotation(
    annotation_dict: dict, 
    db: Session = Depends(get_db)
):
    """Cria uma nova anotação"""
    # Criar instância de AnnotationCreate a partir do dict recebido
    annotation = AnnotationCreate(**annotation_dict)
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == annotation.image_id).first()
    if not image:
        return build_error_response("Imagem não encontrada", 404)
    
    # Se o dataset_id foi fornecido, verificar se existe
    if annotation.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == annotation.dataset_id).first()
        if not dataset:
            return build_error_response("Dataset não encontrado", 404)
        
        # Verificar se a classe está definida no dataset
        if annotation.class_name and dataset.classes:
            if annotation.class_name not in dataset.classes:
                # Se a classe não existe no dataset, adicioná-la
                classes = dataset.classes or []
                classes.append(annotation.class_name)
                dataset.classes = classes
                db.commit()
    
    # Criar a anotação
    db_annotation = Annotation(**annotation.dict())
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)
    
    # Converter para esquema de resposta
    response = AnnotationResponse.from_orm(db_annotation)
    return build_response(response)

@router.get("/", response_model=None)
def list_annotations(
    image_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    class_name: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Lista anotações com filtros opcionais"""
    query = db.query(Annotation)
    
    if image_id:
        query = query.filter(Annotation.image_id == image_id)
    
    if dataset_id:
        query = query.filter(Annotation.dataset_id == dataset_id)
    
    if class_name:
        query = query.filter(Annotation.class_name == class_name)
    
    annotations = query.offset(skip).limit(limit).all()
    
    # Converter para esquema de resposta
    response_list = [AnnotationResponse.from_orm(annotation) for annotation in annotations]
    return build_response(response_list)

@router.get("/{annotation_id}", response_model=None)
def get_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Obtém uma anotação específica"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if annotation is None:
        return build_error_response("Anotação não encontrada", 404)
    
    # Converter para esquema de resposta
    response = AnnotationResponse.from_orm(annotation)
    return build_response(response)

@router.put("/{annotation_id}", response_model=None)
def update_annotation(
    annotation_id: int, 
    annotation_dict: dict, 
    db: Session = Depends(get_db)
):
    """Atualiza uma anotação existente"""
    db_annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if db_annotation is None:
        return build_error_response("Anotação não encontrada", 404)
    
    # Criar instância de AnnotationUpdate a partir do dict recebido
    annotation = AnnotationUpdate(**annotation_dict)
    
    # Se estiver atualizando a classe, verificar se ela existe no dataset
    if hasattr(annotation, 'class_name') and annotation.class_name is not None:
        dataset_id = db_annotation.dataset_id
        if dataset_id:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.classes:
                if annotation.class_name not in dataset.classes:
                    # Se a classe não existe no dataset, adicioná-la
                    classes = dataset.classes or []
                    classes.append(annotation.class_name)
                    dataset.classes = classes
                    db.commit()
    
    # Atualizar os campos da anotação
    annotation_data = annotation.dict(exclude_unset=True)
    for key, value in annotation_data.items():
        setattr(db_annotation, key, value)
    
    db.commit()
    db.refresh(db_annotation)
    
    # Converter para esquema de resposta
    response = AnnotationResponse.from_orm(db_annotation)
    return build_response(response)

@router.delete("/{annotation_id}", response_model=None)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Remove uma anotação"""
    db_annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if db_annotation is None:
        return build_error_response("Anotação não encontrada", 404)
    
    db.delete(db_annotation)
    db.commit()
    
    return build_response({"message": "Anotação removida com sucesso"})

@router.post("/batch", response_model=None)
def create_annotations_batch(
    annotations_dict: dict, 
    db: Session = Depends(get_db)
):
    """Cria múltiplas anotações em lote"""
    # Extrair image_id e dataset_id diretamente do dict
    image_id = annotations_dict.get("image_id")
    dataset_id = annotations_dict.get("dataset_id")
    annotations_list = annotations_dict.get("annotations", [])
    
    if not image_id:
        return build_error_response("image_id é obrigatório", 400)
    
    results = []
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        return build_error_response("Imagem não encontrada", 404)
    
    # Se o dataset_id foi fornecido, verificar se existe
    dataset = None
    if dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return build_error_response("Dataset não encontrado", 404)
    
    # Processar cada anotação do lote
    for annotation_data in annotations_list:
        # Adicionar image_id e dataset_id a cada anotação
        annotation_data["image_id"] = image_id
        if dataset_id:
            annotation_data["dataset_id"] = dataset_id
        
        # Verificar se a classe está definida no dataset
        if dataset and annotation_data.get("class_name"):
            class_name = annotation_data["class_name"]
            if dataset.classes and class_name not in dataset.classes:
                # Se a classe não existe no dataset, adicioná-la
                classes = dataset.classes or []
                classes.append(class_name)
                dataset.classes = classes
                db.commit()
        
        # Criar a anotação
        db_annotation = Annotation(**annotation_data)
        db.add(db_annotation)
        results.append(db_annotation)
    
    db.commit()
    
    # Atualizar os objetos com os IDs gerados
    for annotation in results:
        db.refresh(annotation)
    
    # Converter para esquema de resposta
    response_list = [AnnotationResponse.from_orm(annotation) for annotation in results]
    return build_response(response_list)

@router.get("/dataset/{dataset_id}/classes", response_model=None)
def get_dataset_classes(dataset_id: int, db: Session = Depends(get_db)):
    """Obtém todas as classes definidas em um dataset e suas contagens"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        return build_error_response("Dataset não encontrado", 404)
    
    # Obter a lista de classes definidas no dataset
    defined_classes = dataset.classes or []
    
    # Contar anotações por classe
    class_query = text("""
        SELECT a.class_name, COUNT(a.id) as count
        FROM annotations a
        JOIN images i ON i.id = a.image_id
        WHERE (i.dataset_id = :dataset_id OR 
            EXISTS (SELECT 1 FROM dataset_images di WHERE di.dataset_id = :dataset_id AND di.image_id = i.id))
            AND a.class_name IS NOT NULL
        GROUP BY a.class_name
        ORDER BY count DESC
    """)
    class_counts = db.execute(class_query, {"dataset_id": dataset_id}).fetchall()
    
    # Formatar o resultado
    result = []
    total_annotations = sum(count for _, count in class_counts)
    class_count_dict = {class_name: count for class_name, count in class_counts}
    
    # Adicionar classes definidas no dataset
    for class_name in defined_classes:
        count = class_count_dict.get(class_name, 0)
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        result.append({
            "class_name": class_name,
            "count": count,
            "percentage": percentage,
            "is_defined": True
        })
    
    # Adicionar classes usadas em anotações mas não definidas no dataset
    for class_name, count in class_counts:
        if class_name not in defined_classes:
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            result.append({
                "class_name": class_name,
                "count": count,
                "percentage": percentage,
                "is_defined": False
            })
    
    return build_response(result)

@router.post("/dataset/{dataset_id}/export", response_model=None)
async def export_dataset_annotations(
    dataset_id: int,
    export_format: str = "yolo",
    destination: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Exporta as anotações de um dataset para o formato YOLO ou outro formato compatível.
    
    Args:
        dataset_id: ID do dataset
        export_format: Formato de exportação (yolo, coco)
        destination: Diretório de destino (opcional)
    
    Returns:
        Caminho do diretório de exportação
    """
    from microdetect.services.annotation_service import AnnotationService
    from pathlib import Path
    
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    # Verificar se há imagens e anotações
    image_count = db.query(Image).filter(Image.dataset_id == dataset_id).count()
    if image_count == 0:
        return build_error_response("Dataset não contém imagens", 400)
    
    # Criar serviço de anotação
    annotation_service = AnnotationService()
    
    try:
        # Definir diretório de destino, se fornecido
        destination_dir = Path(destination) if destination else None
        
        # Exportar anotações
        export_path = await annotation_service.export_annotations(
            dataset_id=dataset_id,
            export_format=export_format,
            destination_dir=destination_dir
        )
        
        return build_response({
            "message": f"Anotações exportadas com sucesso para o formato {export_format}",
            "export_path": export_path,
            "export_format": export_format,
            "dataset_id": dataset_id,
            "image_count": image_count
        })
    
    except Exception as e:
        return build_error_response(f"Erro ao exportar anotações: {str(e)}", 500)

@router.post("/dataset/{dataset_id}/import", response_model=None)
async def import_dataset_annotations(
    dataset_id: int,
    import_format: str = "yolo",
    source_dir: str = None,
    db: Session = Depends(get_db)
):
    """
    Importa anotações para um dataset a partir de arquivos no formato YOLO ou outro formato compatível.
    
    Args:
        dataset_id: ID do dataset
        import_format: Formato das anotações (yolo, coco)
        source_dir: Diretório contendo as anotações
    
    Returns:
        Número de anotações importadas
    """
    from microdetect.services.annotation_service import AnnotationService
    from pathlib import Path
    
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    if not source_dir:
        return build_error_response("Diretório de origem não especificado", 400)
    
    # Verificar se o diretório existe
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        return build_error_response(f"Diretório não encontrado: {source_dir}", 404)
    
    # Criar serviço de anotação
    annotation_service = AnnotationService()
    
    try:
        # Importar anotações
        count = await annotation_service.import_annotations(
            dataset_id=dataset_id,
            import_format=import_format,
            source_dir=source_path
        )
        
        return build_response({
            "message": f"Importação concluída com sucesso",
            "annotations_imported": count,
            "import_format": import_format,
            "dataset_id": dataset_id
        })
    
    except Exception as e:
        return build_error_response(f"Erro ao importar anotações: {str(e)}", 500)

@router.post("/dataset/{dataset_id}/convert-to-yolo", response_model=None)
async def convert_annotations_to_yolo(
    dataset_id: int,
    destination: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Converte as anotações de um dataset para o formato YOLO e prepara a estrutura 
    de diretórios para treinamento seguindo o padrão YOLO.
    
    A estrutura será criada conforme abaixo:
    - ~/.microdetect/data/training/nome_do_dataset/
      - images/
        - train/
        - val/
        - test/
      - labels/
        - train/
        - val/
        - test/
      - data.yaml
    
    Args:
        dataset_id: ID do dataset
        destination: Diretório de destino (opcional)
    
    Returns:
        Informações sobre a exportação e caminho do diretório de treinamento
    """
    from microdetect.services.annotation_service import AnnotationService
    from pathlib import Path
    
    # Verificar se o dataset existe
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return build_error_response("Dataset não encontrado", 404)
    
    # Verificar se há imagens
    image_count = db.query(Image).filter(Image.dataset_id == dataset_id).count()
    if image_count == 0:
        return build_error_response("Dataset não contém imagens", 400)
    
    # Verificar se há anotações
    annotation_count = db.query(Annotation).join(Image).filter(
        (Image.dataset_id == dataset_id) | 
        (Annotation.dataset_id == dataset_id)
    ).count()
    
    if annotation_count == 0:
        return build_error_response("Dataset não contém anotações", 400)
    
    # Criar serviço de anotação
    annotation_service = AnnotationService()
    
    try:
        # Definir diretório de destino, se fornecido
        destination_dir = Path(destination) if destination else None
        
        # Exportar anotações
        export_path = await annotation_service.export_annotations(
            dataset_id=dataset_id,
            export_format="yolo",
            destination_dir=destination_dir
        )
        
        return build_response({
            "message": "Dataset convertido com sucesso para o formato YOLO",
            "export_path": export_path,
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "image_count": image_count,
            "annotation_count": annotation_count,
            "classes": dataset.classes
        })
    
    except Exception as e:
        return build_error_response(f"Erro ao converter dataset para YOLO: {str(e)}", 500) 
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
    try:
        # Extrair coordenadas, se fornecidas diretamente
        coords = {}
        for field in ['x', 'y', 'width', 'height']:
            if field in annotation_dict:
                coords[field] = annotation_dict[field]
        
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
        
        # Preparar os dados para o modelo Annotation
        # Campos válidos para o modelo Annotation (sem bounding_box, que é processado abaixo)
        valid_fields = ['image_id', 'dataset_id', 'class_name', 'confidence', 'metadata']
        
        # Filtrar apenas campos válidos para evitar erro com campos extras (como class_id)
        filtered_data = {k: v for k, v in annotation.dict().items() if k in valid_fields}
        
        # Processar bounding_box
        if hasattr(annotation, 'bounding_box') and annotation.bounding_box:
            bbox = annotation.bounding_box
            filtered_data['bbox'] = bbox
            
            # Atualizar coordenadas do bounding_box
            filtered_data['x'] = bbox.get('x')
            filtered_data['y'] = bbox.get('y')
            filtered_data['width'] = bbox.get('width')
            filtered_data['height'] = bbox.get('height')
        else:
            # Se não tiver bounding_box mas tiver coordenadas diretas, usar elas
            for field in ['x', 'y', 'width', 'height']:
                if field in coords:
                    filtered_data[field] = coords[field]
            
            # Criar bbox com base nas coordenadas diretas
            if all(field in filtered_data for field in ['x', 'y', 'width', 'height']):
                filtered_data['bbox'] = {
                    'x': filtered_data['x'],
                    'y': filtered_data['y'],
                    'width': filtered_data['width'],
                    'height': filtered_data['height']
                }
        
        # Calcular área automaticamente
        if 'width' in filtered_data and 'height' in filtered_data:
            filtered_data['area'] = filtered_data['width'] * filtered_data['height']
        
        # Criar a anotação
        db_annotation = Annotation(**filtered_data)
        db.add(db_annotation)
        db.commit()
        db.refresh(db_annotation)
        
        # Criar uma resposta simplificada sem referências circulares
        simplified_response = {
            "id": db_annotation.id,
            "image_id": db_annotation.image_id,
            "dataset_id": db_annotation.dataset_id,
            "class_name": db_annotation.class_name,
            "confidence": db_annotation.confidence,
            "x": db_annotation.x,
            "y": db_annotation.y,
            "width": db_annotation.width,
            "height": db_annotation.height,
            "area": db_annotation.area,
            "bbox": db_annotation.bbox,
            "created_at": db_annotation.created_at.isoformat() if db_annotation.created_at else None,
            "updated_at": db_annotation.updated_at.isoformat() if db_annotation.updated_at else None
        }
        
        return build_response(simplified_response)
    except Exception as e:
        return build_error_response(f"Erro ao criar anotação: {str(e)}", 500)

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
    try:
        query = db.query(Annotation)
        
        if image_id:
            query = query.filter(Annotation.image_id == image_id)
        
        if dataset_id:
            query = query.filter(Annotation.dataset_id == dataset_id)
        
        if class_name:
            query = query.filter(Annotation.class_name == class_name)
        
        annotations = query.offset(skip).limit(limit).all()
        
        # Criar resposta simplificada para cada anotação
        simplified_annotations = []
        for annotation in annotations:
            simplified_annotations.append({
                "id": annotation.id,
                "image_id": annotation.image_id,
                "dataset_id": annotation.dataset_id,
                "class_name": annotation.class_name,
                "confidence": annotation.confidence,
                "x": annotation.x,
                "y": annotation.y,
                "width": annotation.width,
                "height": annotation.height,
                "area": annotation.area,
                "bbox": annotation.bbox,
                "created_at": annotation.created_at.isoformat() if annotation.created_at else None,
                "updated_at": annotation.updated_at.isoformat() if annotation.updated_at else None
            })
        
        return build_response(simplified_annotations)
    except Exception as e:
        return build_error_response(f"Erro ao listar anotações: {str(e)}", 500)

@router.get("/{annotation_id}", response_model=None)
def get_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Obtém uma anotação específica"""
    try:
        annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
        if annotation is None:
            return build_error_response("Anotação não encontrada", 404)
        
        # Criar uma resposta simplificada sem referências circulares
        simplified_response = {
            "id": annotation.id,
            "image_id": annotation.image_id,
            "dataset_id": annotation.dataset_id,
            "class_name": annotation.class_name,
            "confidence": annotation.confidence,
            "x": annotation.x,
            "y": annotation.y,
            "width": annotation.width,
            "height": annotation.height,
            "area": annotation.area,
            "bbox": annotation.bbox,
            "created_at": annotation.created_at.isoformat() if annotation.created_at else None,
            "updated_at": annotation.updated_at.isoformat() if annotation.updated_at else None
        }
        
        return build_response(simplified_response)
    except Exception as e:
        return build_error_response(f"Erro ao buscar anotação: {str(e)}", 500)

@router.put("/{annotation_id}", response_model=None)
def update_annotation(
    annotation_id: int, 
    annotation_dict: dict, 
    db: Session = Depends(get_db)
):
    """Atualiza uma anotação existente"""
    try:
        db_annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
        if db_annotation is None:
            return build_error_response("Anotação não encontrada", 404)
        
        # Extrair coordenadas e dimensões, se fornecidas
        coords = {}
        for field in ['x', 'y', 'width', 'height']:
            if field in annotation_dict:
                coords[field] = annotation_dict[field]
        
        # Extrair bounding_box separadamente, se existir
        bounding_box = None
        if 'bounding_box' in annotation_dict:
            bounding_box = annotation_dict['bounding_box']
            # Extrair coordenadas do bounding_box se não fornecidas diretamente
            for field in ['x', 'y', 'width', 'height']:
                if field not in coords and field in bounding_box:
                    coords[field] = bounding_box[field]
        
        # Remover campos que não são aceitos pelo schema de atualização
        fields_to_remove = ['id', 'x', 'y', 'width', 'height', 'bounding_box', 'area']
        for field in fields_to_remove:
            if field in annotation_dict:
                del annotation_dict[field]
        
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
        
        # Atualizar os campos da anotação a partir do schema
        annotation_data = annotation.dict(exclude_unset=True)
        for key, value in annotation_data.items():
            setattr(db_annotation, key, value)
        
        # Atualizar coordenadas e dimensões, se fornecidas
        if coords:
            if 'x' in coords:
                db_annotation.x = coords['x']
            if 'y' in coords:
                db_annotation.y = coords['y']
            if 'width' in coords:
                db_annotation.width = coords['width']
            if 'height' in coords:
                db_annotation.height = coords['height']
            
            # Calcular área automaticamente se width e height estiverem disponíveis
            if db_annotation.width is not None and db_annotation.height is not None:
                db_annotation.area = db_annotation.width * db_annotation.height
            
            # Atualizar o bbox com base nas coordenadas atualizadas
            db_annotation.bbox = {
                'x': db_annotation.x,
                'y': db_annotation.y,
                'width': db_annotation.width,
                'height': db_annotation.height
            }
        
        db.commit()
        db.refresh(db_annotation)
        
        # Criar uma resposta simplificada sem referências circulares
        simplified_response = {
            "id": db_annotation.id,
            "image_id": db_annotation.image_id,
            "dataset_id": db_annotation.dataset_id,
            "class_name": db_annotation.class_name,
            "confidence": db_annotation.confidence,
            "x": db_annotation.x,
            "y": db_annotation.y,
            "width": db_annotation.width,
            "height": db_annotation.height,
            "area": db_annotation.area,
            "bbox": db_annotation.bbox,
            "created_at": db_annotation.created_at.isoformat() if db_annotation.created_at else None,
            "updated_at": db_annotation.updated_at.isoformat() if db_annotation.updated_at else None
        }
        
        return build_response(simplified_response)
    except Exception as e:
        return build_error_response(f"Erro ao atualizar anotação: {str(e)}", 500)

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
    
    # Campos válidos para o modelo Annotation
    valid_fields = ['image_id', 'dataset_id', 'class_name', 'x', 'y', 'width', 'height', 'confidence', 'source', 'metadata']
    
    # Processar cada anotação do lote
    for annotation_data in annotations_list:
        # Filtrar apenas campos válidos para evitar erro com campos extras (como class_id)
        filtered_data = {k: v for k, v in annotation_data.items() if k in valid_fields}
        
        # Adicionar image_id e dataset_id a cada anotação
        filtered_data["image_id"] = image_id
        if dataset_id:
            filtered_data["dataset_id"] = dataset_id
        
        # Verificar se a classe está definida no dataset
        if dataset and filtered_data.get("class_name"):
            class_name = filtered_data["class_name"]
            if dataset.classes and class_name not in dataset.classes:
                # Se a classe não existe no dataset, adicioná-la
                classes = dataset.classes or []
                classes.append(class_name)
                dataset.classes = classes
                db.commit()
        
        # Criar bbox com base nas coordenadas, se não fornecido
        if all(field in filtered_data for field in ['x', 'y', 'width', 'height']):
            filtered_data['bbox'] = {
                'x': filtered_data['x'],
                'y': filtered_data['y'],
                'width': filtered_data['width'],
                'height': filtered_data['height']
            }
            
            # Calcular área automaticamente
            filtered_data['area'] = filtered_data['width'] * filtered_data['height']
        
        # Criar a anotação
        db_annotation = Annotation(**filtered_data)
        db.add(db_annotation)
        results.append(db_annotation)
    
    db.commit()
    
    # Atualizar os objetos com os IDs gerados
    for annotation in results:
        db.refresh(annotation)
    
    # Em vez de retornar os objetos completos, retornar apenas os IDs e informação básica
    # para evitar problemas de serialização
    response_data = {
        "success": True,
        "count": len(results),
        "message": f"{len(results)} anotações criadas com sucesso",
        "annotation_ids": [annotation.id for annotation in results]
    }
    
    return build_response(response_data)

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
        JOIN dataset_images di ON di.image_id = i.id
        WHERE di.dataset_id = :dataset_id
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
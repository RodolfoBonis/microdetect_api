from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from microdetect.database.database import get_db
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset
from microdetect.models.dataset_image import DatasetImage
from microdetect.schemas.image import ImageResponse, ImageUpdate
from microdetect.services.image_service import ImageService
from microdetect.utils.serializers import build_response, build_error_response
from sqlalchemy import or_

router = APIRouter()
image_service = ImageService()

@router.post("/", response_model=None)
async def upload_image(
    file: UploadFile = File(...),
    dataset_id: Optional[int] = Form(None),
    metadata: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Faz upload de uma imagem."""
    # Validar tipo de arquivo
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
        raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
    
    # Ler conteúdo do arquivo
    content = await file.read()
    
    # Processar metadados
    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, 
                detail="Formato de metadados inválido. Deve ser um JSON válido."
            )
    
    # Log para debug
    print(f"Recebido upload com dataset_id: {dataset_id}")
    
    # Salvar imagem usando o serviço
    image_info = image_service.save_image(
        image_data=content,
        filename=file.filename,
        dataset_id=dataset_id,  # Passar explicitamente o dataset_id
        metadata=parsed_metadata,
        width=width,
        height=height
    )
    
    # Adicionar dimensões da imagem se fornecidas
    if width is not None:
        image_info['width'] = width
    if height is not None:
        image_info['height'] = height
    
    # Garantir que temos metadados estruturados (não apenas como string)
    if 'image_metadata' in image_info and isinstance(image_info['image_metadata'], dict):
        # Incluir dimensões nos metadados também
        if width is not None and 'width' not in image_info['image_metadata']:
            image_info['image_metadata']['width'] = width
        if height is not None and 'height' not in image_info['image_metadata']:
            image_info['image_metadata']['height'] = height
    
    # Criar registro no banco
    # Remover dataset_id do dicionário antes de criar o objeto Image
    if 'dataset_id' in image_info:
        dataset_id = image_info.pop('dataset_id')
    
    db_image = Image(**image_info)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    # Se temos um dataset_id, criar a associação entre a imagem e o dataset
    if dataset_id is not None:
        try:
            # Verificar se o dataset existe
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                # Criar associação na tabela de relacionamento
                dataset_image = DatasetImage(
                    dataset_id=dataset_id,
                    image_id=db_image.id,
                    path=db_image.file_path  # Adicionar o caminho do arquivo
                )
                db.add(dataset_image)
                db.commit()
                
                # Adicionar informação sobre o dataset na resposta
                if not hasattr(db_image, 'datasets'):
                    db_image.datasets = []
                db_image.datasets.append(dataset)
        except Exception as e:
            print(f"Erro ao associar imagem ao dataset: {e}")
            # Não falhar o upload caso a associação dê erro
    
    # Converter o modelo ORM para a classe de resposta
    response = ImageResponse.from_orm(db_image)
    return build_response(response)

@router.get("/", response_model=None)
def list_images(
    dataset_id: Optional[int] = None,
    with_annotations: bool = False,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista todas as imagens com seus datasets associados e opcionalmente suas anotações."""
    # Criar query base
    query = db.query(Image)
    
    # Filtrar por dataset_id se fornecido
    if dataset_id:
        # Buscar imagens que pertencem ao dataset específico através da relação N:N
        query = query.outerjoin(DatasetImage, Image.id == DatasetImage.image_id).filter(
            DatasetImage.dataset_id == dataset_id
        ).distinct()
    
    # Executar a query com paginação
    images = query.offset(skip).limit(limit).all()
    
    # Lista para armazenar os resultados simplificados
    simplified_images = []
    
    # Para cada imagem, preparar uma resposta simplificada
    for image in images:
        # Obter as associações através da tabela pivô
        dataset_associations = db.query(DatasetImage).filter(DatasetImage.image_id == image.id).all()
        dataset_ids = [assoc.dataset_id for assoc in dataset_associations]
        
        # Preparar lista simplificada de datasets
        datasets = []
        if dataset_ids:
            # Buscar os datasets correspondentes
            db_datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
            datasets = [
                {
                    "id": d.id,
                    "name": d.name,
                    "description": d.description
                }
                for d in db_datasets
            ]
        
        # Preparar lista simplificada de anotações, se solicitado
        annotations = []
        if with_annotations:
            from microdetect.models.annotation import Annotation
            db_annotations = db.query(Annotation).filter(Annotation.image_id == image.id).all()
            annotations = [
                {
                    "id": a.id,
                    "class_name": a.class_name,
                    "x": a.x,
                    "y": a.y,
                    "width": a.width,
                    "height": a.height,
                    "confidence": a.confidence
                }
                for a in db_annotations
            ]
        
        # Criar um dicionário com os dados simplificados da imagem
        image_data = {
            "id": image.id,
            "file_name": image.file_name,
            "file_path": image.file_path,
            "url": image.url,
            "width": image.width,
            "height": image.height,
            "file_size": image.file_size,
            "created_at": image.created_at.isoformat() if image.created_at else None,
            "updated_at": image.updated_at.isoformat() if image.updated_at else None,
            "datasets": datasets,
            "annotations": annotations
        }
        
        simplified_images.append(image_data)
    
    # Retornar a lista simplificada
    return build_response(simplified_images)

@router.get("/{image_id}", response_model=None)
def get_image(
    image_id: int, 
    with_annotations: bool = True,
    db: Session = Depends(get_db)
):
    """Obtém uma imagem específica com seus datasets associados e opcionalmente suas anotações."""
    # Buscar a imagem pelo ID
    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        return build_error_response("Imagem não encontrada", 404)
    
    # Obter as associações através da tabela pivô
    dataset_associations = db.query(DatasetImage).filter(DatasetImage.image_id == image.id).all()
    dataset_ids = [assoc.dataset_id for assoc in dataset_associations]
    
    # Preparar lista simplificada de datasets
    datasets = []
    if dataset_ids:
        # Buscar os datasets correspondentes
        db_datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
        datasets = [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description
            }
            for d in db_datasets
        ]
    
    # Preparar lista simplificada de anotações, se solicitado
    annotations = []
    if with_annotations:
        from microdetect.models.annotation import Annotation
        db_annotations = db.query(Annotation).filter(Annotation.image_id == image.id).all()
        annotations = [
            {
                "id": a.id,
                "class_name": a.class_name,
                "x": a.x,
                "y": a.y,
                "width": a.width,
                "height": a.height,
                "confidence": a.confidence
            }
            for a in db_annotations
        ]
    
    # Criar um dicionário com os dados simplificados da imagem
    image_data = {
        "id": image.id,
        "file_name": image.file_name,
        "file_path": image.file_path,
        "url": image.url,
        "width": image.width,
        "height": image.height,
        "file_size": image.file_size,
        "created_at": image.created_at.isoformat() if image.created_at else None,
        "updated_at": image.updated_at.isoformat() if image.updated_at else None,
        "datasets": datasets,
        "annotations": annotations
    }
    
    # Retornar o dicionário simplificado
    return build_response(image_data)

@router.put("/{image_id}", response_model=None)
def update_image(
    image_id: int,
    image_update_dict: dict,
    db: Session = Depends(get_db)
):
    """Atualiza uma imagem existente."""
    db_image = db.query(Image).filter(Image.id == image_id).first()
    if db_image is None:
        return build_error_response("Imagem não encontrada", 404)
    
    try:
        # Criar instância de ImageUpdate a partir do dict recebido
        image_update = ImageUpdate(**image_update_dict)
        
        for key, value in image_update.dict(exclude_unset=True).items():
            setattr(db_image, key, value)
        
        db.commit()
        db.refresh(db_image)
        
        # Converter o modelo ORM para a classe de resposta
        response = ImageResponse.from_orm(db_image)
        return build_response(response)
    except Exception as e:
        return build_error_response(f"Erro ao atualizar imagem: {str(e)}", 500)

@router.delete("/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Remove uma imagem."""
    db_image = db.query(Image).filter(Image.id == image_id).first()
    if db_image is None:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Remover arquivo físico
    if not image_service.delete_image(db_image.file_name, db_image.dataset_id):
        raise HTTPException(status_code=500, detail="Erro ao remover arquivo físico")
    
    # Remover associações com datasets
    db.query(DatasetImage).filter(DatasetImage.image_id == image_id).delete()
    
    # Remover registro do banco
    db.delete(db_image)
    db.commit()
    
    return build_response({"message": "Imagem removida com sucesso"})

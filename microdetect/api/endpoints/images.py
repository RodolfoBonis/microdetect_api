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
from sqlalchemy import or_

router = APIRouter()
image_service = ImageService()

@router.post("/", response_model=ImageResponse)
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
                    image_id=db_image.id
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
    
    return db_image

@router.get("/", response_model=List[ImageResponse])
def list_images(
    dataset_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista todas as imagens com seus datasets associados."""
    # Criar query base
    query = db.query(Image)
    
    # Filtrar por dataset_id se fornecido
    if dataset_id:
        # Buscar imagens que pertencem ao dataset específico (tanto pela coluna dataset_id quanto pela relação N:N)
        query = query.outerjoin(DatasetImage, Image.id == DatasetImage.image_id).filter(
            or_(Image.dataset_id == dataset_id, DatasetImage.dataset_id == dataset_id)
        ).distinct()
    
    # Executar a query com paginação
    images = query.offset(skip).limit(limit).all()
    
    # Para cada imagem, buscar os datasets associados
    for image in images:
        # Obter as associações através da tabela pivô
        dataset_associations = db.query(DatasetImage).filter(DatasetImage.image_id == image.id).all()
        dataset_ids = [assoc.dataset_id for assoc in dataset_associations]
        
        # Incluir também o dataset_id primário, se existir
        if image.dataset_id is not None and image.dataset_id not in dataset_ids:
            dataset_ids.append(image.dataset_id)
            
        # Se não há datasets associados, continuar para a próxima imagem
        if not dataset_ids:
            image.datasets = []
            continue
            
        # Buscar os datasets correspondentes
        datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
        
        # Atribuir à propriedade datasets da imagem
        image.datasets = datasets
    
    return images

@router.get("/{image_id}", response_model=ImageResponse)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """Obtém uma imagem específica com seus datasets associados."""
    # Buscar a imagem pelo ID
    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Obter as associações através da tabela pivô
    dataset_associations = db.query(DatasetImage).filter(DatasetImage.image_id == image.id).all()
    dataset_ids = [assoc.dataset_id for assoc in dataset_associations]
    
    # Incluir também o dataset_id primário, se existir
    if image.dataset_id is not None and image.dataset_id not in dataset_ids:
        dataset_ids.append(image.dataset_id)
        
    # Se não há datasets associados, retornar a imagem sem datasets
    if not dataset_ids:
        image.datasets = []
        return image
        
    # Buscar os datasets correspondentes
    datasets = db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
    
    # Atribuir à propriedade datasets da imagem
    image.datasets = datasets
    
    return image

@router.put("/{image_id}", response_model=ImageResponse)
def update_image(
    image_id: int,
    image_update: ImageUpdate,
    db: Session = Depends(get_db)
):
    """Atualiza uma imagem existente."""
    db_image = db.query(Image).filter(Image.id == image_id).first()
    if db_image is None:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    for key, value in image_update.dict(exclude_unset=True).items():
        setattr(db_image, key, value)
    
    db.commit()
    db.refresh(db_image)
    return db_image

@router.delete("/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Remove uma imagem."""
    db_image = db.query(Image).filter(Image.id == image_id).first()
    if db_image is None:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Remover arquivo físico
    if not image_service.delete_image(db_image.file_name, db_image.dataset_id):
        raise HTTPException(status_code=500, detail="Erro ao remover arquivo")
    
    # Remover registro do banco
    db.delete(db_image)
    db.commit()
    
    return {"message": "Imagem removida com sucesso"}

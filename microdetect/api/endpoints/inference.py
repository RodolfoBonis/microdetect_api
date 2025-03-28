from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from microdetect.database.database import get_db
from microdetect.models.inference_result import InferenceResult
from microdetect.models.model import Model
from microdetect.schemas.inference_result import InferenceResultResponse
from microdetect.services.yolo_service import YOLOService
from microdetect.services.image_service import ImageService
from microdetect.utils.serializers import build_response, build_error_response

router = APIRouter()
yolo_service = YOLOService()
image_service = ImageService()

@router.post("/", response_model=None)
async def perform_inference(
    file: UploadFile = File(...),
    model_id: int = None,
    confidence_threshold: float = 0.5,
    db: Session = Depends(get_db)
):
    """Realiza inferência em uma imagem."""
    # Validar modelo
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        return build_error_response("Modelo não encontrado", 404)
    
    # Validar tipo de arquivo
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff"]:
        return build_error_response("Tipo de arquivo não suportado", 400)
    
    # Ler conteúdo do arquivo
    content = await file.read()
    
    # Salvar imagem temporariamente
    image_info = image_service.save_image(
        image_data=content,
        filename=file.filename,
        dataset_id=None
    )
    
    try:
        # Realizar inferência
        predictions, metrics = await yolo_service.predict(
            model_id=model_id,
            image_path=image_info["filepath"],
            confidence_threshold=confidence_threshold
        )
        
        # Criar resultado
        result = InferenceResult(
            predictions=predictions,
            metrics=metrics,
            image_id=image_info["id"],
            model_id=model_id
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        
        # Converter para esquema de resposta
        response = InferenceResultResponse.from_orm(result)
        return build_response(response)
        
    finally:
        # Limpar arquivo temporário
        image_service.delete_image(image_info["filename"])

@router.get("/", response_model=None)
def list_inference_results(
    image_id: int = None,
    model_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista todos os resultados de inferência."""
    query = db.query(InferenceResult)
    if image_id:
        query = query.filter(InferenceResult.image_id == image_id)
    if model_id:
        query = query.filter(InferenceResult.model_id == model_id)
    results = query.offset(skip).limit(limit).all()
    
    # Converter para esquema de resposta
    response_list = [InferenceResultResponse.from_orm(result) for result in results]
    return build_response(response_list)

@router.get("/{result_id}", response_model=None)
def get_inference_result(result_id: int, db: Session = Depends(get_db)):
    """Obtém um resultado de inferência específico."""
    result = db.query(InferenceResult).filter(InferenceResult.id == result_id).first()
    if result is None:
        return build_error_response("Resultado de inferência não encontrado", 404)
    
    # Converter para esquema de resposta
    response = InferenceResultResponse.from_orm(result)
    return build_response(response) 
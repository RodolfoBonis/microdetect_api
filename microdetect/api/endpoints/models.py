from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from microdetect.database.database import get_db
from microdetect.models.model import Model
from microdetect.schemas.model import ModelCreate, ModelResponse, ModelUpdate
from microdetect.utils.serializers import build_response, build_error_response

router = APIRouter()

@router.post("/", response_model=None)
def create_model(model: dict, db: Session = Depends(get_db)):
    """Cria um novo modelo."""
    # Criar instância de ModelCreate a partir do dict recebido
    model_create = ModelCreate(**model)
    
    # Criar registro no banco
    db_model = Model(**model_create.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    # Converter para esquema de resposta
    response = ModelResponse.from_orm(db_model)
    return build_response(response)

@router.get("/", response_model=None)
def list_models(
    training_session_id: int = None,
    model_type: str = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista todos os modelos."""
    query = db.query(Model)
    if training_session_id:
        query = query.filter(Model.training_session_id == training_session_id)
    if model_type:
        query = query.filter(Model.model_type == model_type)
    models = query.offset(skip).limit(limit).all()
    
    # Converter para esquema de resposta
    response_list = [ModelResponse.from_orm(model) for model in models]
    return build_response(response_list)

@router.get("/{model_id}", response_model=None)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Obtém um modelo específico."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if model is None:
        return build_error_response("Modelo não encontrado", 404)
    
    # Converter para esquema de resposta
    response = ModelResponse.from_orm(model)
    return build_response(response)

@router.put("/{model_id}", response_model=None)
def update_model(
    model_id: int,
    model_update_dict: dict,
    db: Session = Depends(get_db)
):
    """Atualiza um modelo existente."""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if db_model is None:
        return build_error_response("Modelo não encontrado", 404)
    
    # Criar instância de ModelUpdate a partir do dict recebido
    model_update = ModelUpdate(**model_update_dict)
    
    for key, value in model_update.dict(exclude_unset=True).items():
        setattr(db_model, key, value)
    
    db.commit()
    db.refresh(db_model)
    
    # Converter para esquema de resposta
    response = ModelResponse.from_orm(db_model)
    return build_response(response)

@router.delete("/{model_id}", response_model=None)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Remove um modelo."""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if db_model is None:
        return build_error_response("Modelo não encontrado", 404)
    
    db.delete(db_model)
    db.commit()
    return build_response({"message": "Modelo removido com sucesso"}) 
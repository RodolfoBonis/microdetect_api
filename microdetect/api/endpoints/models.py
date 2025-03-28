from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from microdetect.database.database import get_db
from microdetect.models.model import Model
from microdetect.schemas.model import ModelCreate, ModelResponse, ModelUpdate

router = APIRouter()

@router.post("/", response_model=ModelResponse)
def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    """Cria um novo modelo."""
    db_model = Model(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.get("/", response_model=List[ModelResponse])
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
    return models

@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Obtém um modelo específico."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    return model

@router.put("/{model_id}", response_model=ModelResponse)
def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Atualiza um modelo existente."""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    
    for key, value in model_update.dict(exclude_unset=True).items():
        setattr(db_model, key, value)
    
    db.commit()
    db.refresh(db_model)
    return db_model

@router.delete("/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Remove um modelo."""
    db_model = db.query(Model).filter(Model.id == model_id).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    
    db.delete(db_model)
    db.commit()
    return {"message": "Modelo removido com sucesso"} 
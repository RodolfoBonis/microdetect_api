from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from microdetect.database.database import get_db
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.schemas.training_session import TrainingSessionCreate, TrainingSessionResponse, TrainingSessionUpdate
from microdetect.services.yolo_service import YOLOService
from microdetect.utils.serializers import build_response, build_error_response

router = APIRouter()
yolo_service = YOLOService()

@router.post("/", response_model=None)
async def create_training_session(
    training: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Cria uma nova sessão de treinamento."""
    # Criar instância de TrainingSessionCreate a partir do dict recebido
    training_create = TrainingSessionCreate(**training)
    
    # Criar registro no banco
    db_training = TrainingSession(**training_create.dict())
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    
    # Iniciar treinamento em background
    background_tasks.add_task(
        train_model,
        db_training.id,
        db
    )
    
    # Converter para esquema de resposta
    response = TrainingSessionResponse.from_orm(db_training)
    return build_response(response)

@router.get("/", response_model=None)
def list_training_sessions(
    dataset_id: int = None,
    status: TrainingStatus = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista todas as sessões de treinamento."""
    query = db.query(TrainingSession)
    if dataset_id:
        query = query.filter(TrainingSession.dataset_id == dataset_id)
    if status:
        query = query.filter(TrainingSession.status == status)
    sessions = query.offset(skip).limit(limit).all()
    
    # Converter para esquema de resposta
    response_list = [TrainingSessionResponse.from_orm(session) for session in sessions]
    return build_response(response_list)

@router.get("/{session_id}", response_model=None)
def get_training_session(session_id: int, db: Session = Depends(get_db)):
    """Obtém uma sessão de treinamento específica."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if session is None:
        return build_error_response("Sessão de treinamento não encontrada", 404)
    
    # Converter para esquema de resposta
    response = TrainingSessionResponse.from_orm(session)
    return build_response(response)

@router.put("/{session_id}", response_model=None)
def update_training_session(
    session_id: int,
    session_update_dict: dict,
    db: Session = Depends(get_db)
):
    """Atualiza uma sessão de treinamento existente."""
    db_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if db_session is None:
        return build_error_response("Sessão de treinamento não encontrada", 404)
    
    # Criar instância de TrainingSessionUpdate a partir do dict recebido
    session_update = TrainingSessionUpdate(**session_update_dict)
    
    for key, value in session_update.dict(exclude_unset=True).items():
        setattr(db_session, key, value)
    
    db.commit()
    db.refresh(db_session)
    
    # Converter para esquema de resposta
    response = TrainingSessionResponse.from_orm(db_session)
    return build_response(response)

@router.delete("/{session_id}", response_model=None)
def delete_training_session(session_id: int, db: Session = Depends(get_db)):
    """Remove uma sessão de treinamento."""
    db_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Sessão de treinamento não encontrada")
    
    db.delete(db_session)
    db.commit()
    return {"message": "Sessão de treinamento removida com sucesso"}

async def train_model(session_id: int, db: Session):
    """Função para treinar o modelo em background."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        return
    
    try:
        # Atualizar status para running
        session.status = TrainingStatus.RUNNING
        session.started_at = datetime.utcnow()
        db.commit()
        
        # Treinar modelo
        metrics = await yolo_service.train(
            dataset_id=session.dataset_id,
            model_type=session.model_type,
            model_version=session.model_version,
            hyperparameters=session.hyperparameters
        )
        
        # Atualizar status e métricas
        session.status = TrainingStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        session.metrics = metrics
        db.commit()
        
    except Exception as e:
        # Atualizar status para failed
        session.status = TrainingStatus.FAILED
        session.completed_at = datetime.utcnow()
        db.commit()
        raise e 
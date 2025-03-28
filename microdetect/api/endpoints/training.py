from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from microdetect.database.database import get_db
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.schemas.training_session import TrainingSessionCreate, TrainingSessionResponse, TrainingSessionUpdate
from microdetect.services.yolo_service import YOLOService

router = APIRouter()
yolo_service = YOLOService()

@router.post("/", response_model=TrainingSessionResponse)
async def create_training_session(
    training: TrainingSessionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Cria uma nova sessão de treinamento."""
    db_training = TrainingSession(**training.dict())
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    
    # Iniciar treinamento em background
    background_tasks.add_task(
        train_model,
        db_training.id,
        db
    )
    
    return db_training

@router.get("/", response_model=List[TrainingSessionResponse])
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
    return sessions

@router.get("/{session_id}", response_model=TrainingSessionResponse)
def get_training_session(session_id: int, db: Session = Depends(get_db)):
    """Obtém uma sessão de treinamento específica."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão de treinamento não encontrada")
    return session

@router.put("/{session_id}", response_model=TrainingSessionResponse)
def update_training_session(
    session_id: int,
    session_update: TrainingSessionUpdate,
    db: Session = Depends(get_db)
):
    """Atualiza uma sessão de treinamento existente."""
    db_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if db_session is None:
        raise HTTPException(status_code=404, detail="Sessão de treinamento não encontrada")
    
    for key, value in session_update.dict(exclude_unset=True).items():
        setattr(db_session, key, value)
    
    db.commit()
    db.refresh(db_session)
    return db_session

@router.delete("/{session_id}")
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
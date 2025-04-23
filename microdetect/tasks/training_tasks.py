import os
import json
import logging
from celery import Task, shared_task
from microdetect.core.celery_app import celery_app
from microdetect.services.yolo_service import YOLOService
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.database.database import SessionLocal
from microdetect.core.training_core import prepare_training_directory, prepare_training_config, update_training_status
from microdetect.services.training_service import TrainingService
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrainingTask(Task):
    _yolo_service = None
    
    @property
    def yolo_service(self):
        if self._yolo_service is None:
            self._yolo_service = YOLOService()
        return self._yolo_service

@shared_task(bind=True)
def run_training_session(self, session_id: int) -> Dict[str, Any]:
    """Executa uma sessão de treinamento."""
    db = SessionLocal()
    try:
        # Obter sessão
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not session:
            raise ValueError(f"Sessão {session_id} não encontrada")
            
        # Atualizar status
        session.status = TrainingStatus.RUNNING
        db.commit()
        
        # Iniciar treinamento
        training_service = TrainingService()
        result = training_service.train(session)
        
        # Atualizar resultado
        session.status = TrainingStatus.COMPLETED
        session.metrics = result.get("metrics", {})
        db.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "metrics": result.get("metrics", {})
        }
        
    except Exception as e:
        logger.error(f"Erro durante treinamento da sessão {session_id}: {str(e)}")
        session.status = TrainingStatus.FAILED
        db.commit()
        raise
        
    finally:
        db.close() 
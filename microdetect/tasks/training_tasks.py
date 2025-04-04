import os
import json
import logging
from celery import Task
from microdetect.core.celery_app import celery_app
from microdetect.services.yolo_service import YOLOService
from microdetect.models.training_session import TrainingSession
from microdetect.core.database import SessionLocal
from microdetect.core.training_core import prepare_training_directory, prepare_training_config, update_training_status

logger = logging.getLogger(__name__)

class TrainingTask(Task):
    _yolo_service = None
    
    @property
    def yolo_service(self):
        if self._yolo_service is None:
            self._yolo_service = YOLOService()
        return self._yolo_service

@celery_app.task(bind=True, base=TrainingTask)
def train_model(self, session_id: int):
    """
    Task Celery para treinar um modelo
    """
    db = SessionLocal()
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise ValueError(f"Sessão de treinamento {session_id} não encontrada")
            
        # Atualizar status
        update_training_status(session, "training", db=db)
        
        # Preparar diretório de treinamento
        train_dir = prepare_training_directory(session, settings.TRAINING_DIR)
        
        # Configurar treinamento
        config = prepare_training_config(session, train_dir, db)
        
        # Iniciar treinamento
        self.yolo_service.train(
            model_path=session.model_path,
            data_yaml=config["data_yaml"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            device=config["device"]
        )
        
        # Atualizar status final
        update_training_status(session, "completed", db=db)
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Treinamento concluído com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro durante treinamento: {str(e)}")
        if session:
            update_training_status(session, "failed", error_message=str(e), db=db)
        raise
        
    finally:
        db.close() 
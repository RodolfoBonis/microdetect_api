import os
import json
import logging
from celery import Task
from microdetect.core.celery_app import celery_app
from microdetect.services.training_service import TrainingService
from microdetect.services.yolo_service import YOLOService
from microdetect.models.training import TrainingSession
from microdetect.core.database import SessionLocal

logger = logging.getLogger(__name__)

class TrainingTask(Task):
    _training_service = None
    _yolo_service = None
    
    @property
    def training_service(self):
        if self._training_service is None:
            self._training_service = TrainingService()
        return self._training_service
    
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
    try:
        db = SessionLocal()
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise ValueError(f"Sessão de treinamento {session_id} não encontrada")
            
        # Atualizar status
        session.status = "training"
        db.commit()
        
        # Preparar diretório de treinamento
        train_dir = self.training_service.prepare_training_directory(session)
        
        # Configurar treinamento
        config = self.training_service.prepare_training_config(session, train_dir)
        
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
        session.status = "completed"
        db.commit()
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Treinamento concluído com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro durante treinamento: {str(e)}")
        if session:
            session.status = "failed"
            session.error_message = str(e)
            db.commit()
        raise
        
    finally:
        db.close() 
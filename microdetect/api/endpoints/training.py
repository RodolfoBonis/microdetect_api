from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime

from microdetect.core.config import Settings
from microdetect.database.database import get_db
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.models.training_report import TrainingReport
from microdetect.schemas.training_session import TrainingSessionCreate, TrainingSessionResponse, TrainingSessionUpdate
from microdetect.schemas.training_report import TrainingReportResponse
from microdetect.schemas.hyperparam_search import TrainingProgress, TrainingMetrics, ResourceUsage
from microdetect.services.yolo_service import YOLOService
from microdetect.services.resource_monitor import ResourceMonitor
from microdetect.utils.serializers import build_response, build_error_response, serialize_to_dict, JSONEncoder
from microdetect.services.training_service import TrainingService

router = APIRouter()
yolo_service = YOLOService()
resource_monitor = ResourceMonitor()

# Armazenamento em memória para progresso de treinamento
training_progress = {}

@router.post("/", response_model=None)
async def create_training_session(
    training: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Cria uma nova sessão de treinamento."""
    try:
        # Instanciar serviço
        training_service = TrainingService()
        
        # Criar sessão de treinamento
        db_training = await training_service.create_training_session(
            dataset_id=training.get("dataset_id"),
            model_type=training.get("model_type", "yolov8"),
            model_version=training.get("model_version", "n"),
            name=training.get("name"),
            description=training.get("description"),
            hyperparameters=training.get("hyperparameters", {})
        )
        
        # Iniciar treinamento em background
        background_tasks.add_task(
            training_service.start_training,
            db_training.id
        )
        
        # Converter para esquema de resposta
        response = TrainingSessionResponse.from_orm(db_training)
        return build_response(response)
    except Exception as e:
        return build_error_response(f"Erro ao criar sessão de treinamento: {str(e)}", 400)

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

@router.get("/{session_id}/report", response_model=None)
def get_training_report(session_id: int, db: Session = Depends(get_db)):
    """Obtém o relatório de treinamento de uma sessão específica."""
    report = db.query(TrainingReport).filter(TrainingReport.training_session_id == session_id).first()
    if report is None:
        return build_error_response("Relatório de treinamento não encontrado", 404)
    
    # Converter para esquema de resposta
    response = TrainingReportResponse.from_orm(report)
    return build_response(response)

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: int,
    db: Session = Depends(get_db)
):
    """Websocket para monitoramento em tempo real de treinamento."""
    await websocket.accept()
    
    try:
        # Criar instância do serviço de treinamento
        training_service = TrainingService()
        
        # Verificar se a sessão existe
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not session:
            error_json = json.dumps({"error": "Sessão de treinamento não encontrada"}, cls=JSONEncoder)
            await websocket.send_text(error_json)
            await websocket.close()
            return
        
        # Configurar heartbeat
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        
        # Obter dados de progresso em tempo real
        progress_data = training_service.get_progress(session_id)
        
        # Enviar estado inicial
        response = TrainingSessionResponse.from_orm(session)
        initial_data = response.dict()
        
        # Adicionar dados de progresso
        initial_data.update({
            "current_epoch": progress_data.get("current_epoch", 0),
            "total_epochs": progress_data.get("total_epochs", session.hyperparameters.get("epochs", 100)),
            "metrics": progress_data.get("metrics", {}),
            "resources": progress_data.get("resources", {}),
            "progress": progress_data
        })

        json_data = json.dumps(initial_data, cls=JSONEncoder)
        await websocket.send_text(json_data)
        
        # Monitorar progresso
        last_update = None
        last_metrics = None
        last_resources = None
        
        while True:
            try:
                # Obter dados de progresso atualizados
                progress_data = training_service.get_progress(session_id)
                
                # Verificar se houve mudanças significativas
                current_metrics = progress_data.get("metrics", {})
                current_resources = progress_data.get("resources", {})
                
                should_update = (
                    last_update != progress_data or
                    last_metrics != current_metrics or
                    last_resources != current_resources
                )
                
                if should_update:
                    last_update = progress_data.copy()
                    last_metrics = current_metrics.copy()
                    last_resources = current_resources.copy()
                    
                    # Atualizar dados da sessão
                    db.refresh(session)
                    
                    # Criar resposta atualizada
                    response = TrainingSessionResponse.from_orm(session)
                    update_data = response.dict()
                    
                    # Adicionar dados de progresso
                    update_data.update({
                        "current_epoch": progress_data.get("current_epoch", 0),
                        "total_epochs": progress_data.get("total_epochs", session.hyperparameters.get("epochs", 100)),
                        "metrics": current_metrics,
                        "resources": current_resources,
                        "progress": progress_data
                    })
                    
                    json_data = json.dumps(update_data, cls=JSONEncoder)
                    await websocket.send_text(json_data)
                
                # Verificar se o treinamento terminou
                if progress_data.get("status") in ["completed", "failed"] or session.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                    # Enviar relatório final se existir
                    report = db.query(TrainingReport).filter(TrainingReport.training_session_id == session_id).first()
                    if report:
                        response = TrainingReportResponse.from_orm(report)
                        json_data = json.dumps({
                            "type": "final_report",
                            "data": response.dict(),
                            "status": session.status
                        }, cls=JSONEncoder)
                        await websocket.send_text(json_data)
                    break
                
                # Aguardar próxima atualização (intervalo menor para mais realtime)
                await asyncio.sleep(0.1)  # 100ms entre atualizações
                
            except Exception as e:
                logger.error(f"Erro durante monitoramento: {str(e)}")
                error_json = json.dumps({"error": str(e)}, cls=JSONEncoder)
                await websocket.send_text(error_json)
                break
    
    except WebSocketDisconnect:
        logger.info(f"Cliente desconectou do websocket de treinamento {session_id}")
    except Exception as e:
        logger.error(f"Erro no websocket de treinamento: {str(e)}")
        try:
            error_json = json.dumps({"error": str(e)}, cls=JSONEncoder)
            await websocket.send_text(error_json)
        except:
            pass
    finally:
        # Cancelar heartbeat
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()
        # Garantir que o websocket seja fechado
        try:
            await websocket.close()
        except:
            pass

async def send_heartbeat(websocket: WebSocket):
    """Envia heartbeat para manter a conexão viva."""
    try:
        while True:
            await asyncio.sleep(30)  # Heartbeat a cada 30 segundos
            await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Erro no heartbeat: {str(e)}")

async def train_model(session_id: int, db: Session):
    """Função para treinar o modelo em background."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        return
    
    # Histórico de métricas para o relatório
    metrics_history = []
    
    # Iniciar monitoramento de recursos
    resource_monitor.start_monitoring(
        interval=2.0,
        callback=lambda usage: update_resource_usage(session_id, usage)
    )
    
    try:
        # Atualizar status para running
        session.status = TrainingStatus.RUNNING
        session.started_at = datetime.utcnow()
        db.commit()
        
        # Configurar diretório de treinamento
        model_dir = Settings.TRAINING_DIR / f"model_{session.id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar callbacks para progresso
        progress_callback = lambda metrics: update_training_progress(
            session_id=session_id,
            metrics=metrics
        )
        
        # Treinar modelo
        metrics = await yolo_service.train(
            dataset_id=session.dataset_id,
            model_type=session.model_type,
            model_version=session.model_version,
            hyperparameters=session.hyperparameters,
            callback=progress_callback
        )
        
        # Atualizar status e métricas
        session.status = TrainingStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        session.metrics = metrics
        db.commit()
        
        # Gerar relatório de treinamento
        await generate_training_report(session_id, db)
        
    except Exception as e:
        # Atualizar status para failed
        session.status = TrainingStatus.FAILED
        session.completed_at = datetime.utcnow()
        db.commit()
        raise e
    finally:
        # Parar monitoramento de recursos
        resource_usage_history = resource_monitor.stop_monitoring()
        
        # Limpar dados de progresso
        if str(session_id) in training_progress:
            del training_progress[str(session_id)]

def update_training_progress(session_id: int, metrics: Dict[str, Any]):
    """Atualiza o progresso de treinamento em memória."""
    # Obter dados de recursos atuais
    resources = resource_monitor.get_current_usage()
    
    # Criar objeto de métricas
    training_metrics = TrainingMetrics(
        epoch=metrics.get("epoch", 0),
        loss=metrics.get("loss", 0.0),
        val_loss=metrics.get("val_loss"),
        map50=metrics.get("map50"),
        map=metrics.get("map"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        resources=resources
    )
    
    # Criar objeto de progresso
    progress = TrainingProgress(
        current_epoch=metrics.get("epoch", 0),
        total_epochs=metrics.get("total_epochs", 100),
        metrics=training_metrics,
        eta_seconds=metrics.get("eta_seconds")
    )
    
    # Armazenar em memória
    training_progress[str(session_id)] = progress

def update_resource_usage(session_id: int, usage: ResourceUsage):
    """Atualiza os dados de uso de recursos no progresso de treinamento."""
    if str(session_id) not in training_progress:
        return
        
    if hasattr(training_progress[str(session_id)], "metrics"):
        training_progress[str(session_id)].metrics.resources = usage

async def generate_training_report(session_id: int, db: Session):
    """Gera um relatório completo sobre o treinamento."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session or session.status != TrainingStatus.COMPLETED:
        return
    
    try:
        # Obter métricas do modelo
        model_metrics = session.metrics if session.metrics else {}
        
        # Obter estatísticas de uso de recursos
        resource_avg = resource_monitor.get_average_usage()
        resource_max = resource_monitor.get_max_usage()
        
        # Obter matriz de confusão
        confusion_matrix = model_metrics.get("confusion_matrix", [])
        
        # Obter métricas por classe
        class_performance = []
        for class_id, metrics in model_metrics.get("class_stats", {}).items():
            class_performance.append({
                "class_id": int(class_id),
                "class_name": metrics.get("name", f"Class {class_id}"),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1", 0.0),
                "support": metrics.get("support", 0),
                "examples_count": metrics.get("count", 0)
            })
        
        # Tamanho do modelo
        model_size_mb = 0
        model_path = Settings.MODELS_DIR / f"{session.model_type}{session.model_version}_{session.id}.pt"
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Criar relatório
        report = TrainingReport(
            training_session_id=session.id,
            model_name=f"{session.model_type}{session.model_version}_{session.id}",
            dataset_id=session.dataset_id,
            metrics_history=training_progress.get(str(session_id), {}).get("metrics_history", []),
            confusion_matrix=confusion_matrix,
            class_performance=class_performance,
            final_metrics=model_metrics,
            resource_usage_avg=resource_avg.dict(),
            resource_usage_max=resource_max.dict(),
            hyperparameters=session.hyperparameters,
            train_images_count=model_metrics.get("train_count", 0),
            val_images_count=model_metrics.get("val_count", 0),
            test_images_count=model_metrics.get("test_count", 0),
            training_time_seconds=int((session.completed_at - session.started_at).total_seconds()),
            model_size_mb=model_size_mb
        )
        
        # Salvar no banco
        db.add(report)
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise e 
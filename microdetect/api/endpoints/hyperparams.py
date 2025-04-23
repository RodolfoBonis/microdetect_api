from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging

from microdetect.database.database import get_db
from microdetect.models.training_session import TrainingStatus
from microdetect.models.hyperparam_search import HyperparamSearch
from microdetect.schemas.hyperparam_search import (
    HyperparamSearchResponse,
)
from microdetect.services.hyperparam_service import HyperparamService
from microdetect.utils.serializers import build_response, build_error_response, serialize_to_dict, JSONEncoder

router = APIRouter()
hyperparam_service = HyperparamService()
logger = logging.getLogger(__name__)

@router.post("/", response_model=None)
async def create_hyperparam_search(
    search_data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Cria uma nova busca de hiperparâmetros."""
    try:
        # Criar busca no banco
        search = await hyperparam_service.create_hyperparam_session(
            dataset_id=search_data.get("dataset_id"),
            model_type=search_data.get("model_type"),
            model_version=search_data.get("model_version"),
            name=search_data.get("name"),
            description=search_data.get("description"),
            search_space=search_data.get("search_space"),
            max_trials=search_data.get("max_trials", 10)
        )
        
        # Iniciar busca em background
        background_tasks.add_task(
            hyperparam_service.start_hyperparam_search,
            search.id
        )
        
        # Converter para esquema de resposta
        response = HyperparamSearchResponse.from_orm(search)
        return build_response(response)
    except Exception as e:
        return build_error_response(str(e), 400)

@router.get("/", response_model=None)
async def list_hyperparam_searches(
    dataset_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lista buscas de hiperparâmetros com filtros opcionais."""
    searches = await hyperparam_service.list_searches(
        dataset_id=dataset_id,
        status=status,
        skip=skip,
        limit=limit,
        db=db
    )
    
    # Converter para esquema de resposta
    response_list = [HyperparamSearchResponse.from_orm(search) for search in searches]
    return build_response(response_list)

@router.get("/{search_id}", response_model=None)
async def get_hyperparam_search(
    search_id: int,
    db: Session = Depends(get_db)
):
    """Obtém uma busca de hiperparâmetros específica."""
    search = await hyperparam_service.get_search(search_id, db)
    if not search:
        return build_error_response("Busca de hiperparâmetros não encontrada", 404)
    
    # Converter para esquema de resposta
    response = HyperparamSearchResponse.from_orm(search)
    return build_response(response)

@router.delete("/{search_id}", response_model=None)
async def delete_hyperparam_search(
    search_id: int,
    db: Session = Depends(get_db)
):
    """Remove uma busca de hiperparâmetros."""
    deleted = await hyperparam_service.delete_search(search_id, db)
    if not deleted:
        return build_error_response("Busca de hiperparâmetros não encontrada", 404)
    
    return {"message": "Busca de hiperparâmetros removida com sucesso"}

@router.websocket("/ws/{search_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    search_id: int,
    db: Session = Depends(get_db)
):
    """Websocket para monitoramento em tempo real de busca de hiperparâmetros."""
    heartbeat_task = None
    try:
        logger.info(f"WebSocket: Iniciando conexão para search_id={search_id}")
        await websocket.accept()
        logger.info(f"WebSocket: Conexão aceita para search_id={search_id}")
        
        # Verificar se a busca existe
        search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
        if not search:
            logger.warning(f"WebSocket: Busca não encontrada para search_id={search_id}")
            error_json = json.dumps({"error": "Busca não encontrada"}, cls=JSONEncoder)
            await websocket.send_text(error_json)
            await websocket.close(code=1000)  # Fechamento normal
            return
        
        logger.info(f"WebSocket: Busca encontrada para search_id={search_id}, status={search.status}")
        
        # Configurar heartbeat
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        logger.info(f"WebSocket: Heartbeat iniciado para search_id={search_id}")
        
        # Obter dados de progresso em tempo real
        try:
            progress_data = hyperparam_service.get_progress(search_id)
            logger.info(f"WebSocket: Dados de progresso obtidos para search_id={search_id}")
        except Exception as e:
            logger.error(f"WebSocket: Erro ao obter dados de progresso para search_id={search_id}: {str(e)}")
            progress_data = {
                "status": search.status,
                "trials": [],
                "best_params": {},
                "best_metrics": {},
                "current_iteration": 0,
                "iterations_completed": 0,
                "total_iterations": search.iterations
            }
        
        # Enviar estado inicial
        try:
            response = HyperparamSearchResponse.from_orm(search)
            initial_data = response.dict()
            
            # Adaptar os dados para o frontend
            initial_data.update({
                "trials_data": progress_data.get("trials", []),
                "current_iteration": progress_data.get("current_iteration", 0),
                "iterations_completed": progress_data.get("iterations_completed", 0),
                "best_params": progress_data.get("best_params", {}),
                "best_metrics": progress_data.get("best_metrics", {}),
                "progress": progress_data
            })

            json_data = json.dumps(initial_data, cls=JSONEncoder)
            await websocket.send_text(json_data)
            logger.info(f"WebSocket: Estado inicial enviado para search_id={search_id}")
        except Exception as e:
            logger.error(f"WebSocket: Erro ao enviar estado inicial para search_id={search_id}: {str(e)}")
            error_json = json.dumps({"error": f"Erro ao processar dados: {str(e)}"}, cls=JSONEncoder)
            await websocket.send_text(error_json)
            await websocket.close(code=1011)  # Erro interno
            return
        
        # Monitorar atualizações
        last_update = None
        last_trials = None
        last_best_params = None
        last_best_metrics = None
        
        while True:
            try:
                # Verificar se a conexão ainda está ativa
                if websocket.client_state.DISCONNECTED:
                    logger.info(f"WebSocket: Cliente desconectado para search_id={search_id}")
                    break
                    
                # Obter dados de progresso atualizados usando a sessão do parâmetro
                search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
                if not search:
                    logger.warning(f"WebSocket: Busca não encontrada durante monitoramento para search_id={search_id}")
                    break
                    
                progress_data = {
                    "status": search.status,
                    "trials": search.trials_data or [],
                    "best_params": search.best_params or {},
                    "best_metrics": search.best_metrics or {},
                    "current_iteration": len(search.trials_data or []),
                    "iterations_completed": len(search.trials_data or []),
                    "total_iterations": search.iterations
                }
                
                # Verificar se houve mudanças significativas
                current_trials = progress_data.get("trials", [])
                current_best_params = progress_data.get("best_params", {})
                current_best_metrics = progress_data.get("best_metrics", {})
                
                should_update = (
                    last_update != progress_data or
                    last_trials != current_trials or
                    last_best_params != current_best_params or
                    last_best_metrics != current_best_metrics
                )
                
                if should_update:
                    last_update = progress_data.copy()
                    last_trials = current_trials.copy()
                    last_best_params = current_best_params.copy()
                    last_best_metrics = current_best_metrics.copy()
                    
                    try:
                        # Atualizar busca para ter os dados mais recentes
                        response = HyperparamSearchResponse.from_orm(search)
                        update_data = response.dict()
                        
                        # Adaptar os dados para o frontend
                        update_data.update({
                            "trials_data": current_trials,
                            "current_iteration": len(current_trials),
                            "iterations_completed": len(current_trials),
                            "best_params": current_best_params,
                            "best_metrics": current_best_metrics,
                            "progress": progress_data
                        })

                        json_data = json.dumps(update_data, cls=JSONEncoder)
                        await websocket.send_text(json_data)
                        logger.debug(f"WebSocket: Atualização enviada para search_id={search_id}, status={search.status}")
                    except Exception as e:
                        logger.error(f"WebSocket: Erro ao enviar atualização para search_id={search_id}: {str(e)}")
                
                # Verificar se a busca terminou
                if search.status in ["completed", "failed"]:
                    logger.info(f"WebSocket: Busca finalizada para search_id={search_id}, status={search.status}")
                    break
                
                await asyncio.sleep(1)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket: Cliente desconectado durante monitoramento para search_id={search_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket: Erro durante monitoramento para search_id={search_id}: {str(e)}")
                break
        
        # Fechar a conexão normalmente
        logger.info(f"WebSocket: Fechando conexão normalmente para search_id={search_id}")
        await websocket.close(code=1000)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket: Cliente desconectado para search_id={search_id}")
    except Exception as e:
        logger.error(f"WebSocket: Erro durante monitoramento para search_id={search_id}: {str(e)}")
        try:
            await websocket.close(code=1011)  # Erro interno
        except:
            pass
    finally:
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info(f"WebSocket: Conexão finalizada para search_id={search_id}")

async def send_heartbeat(websocket: WebSocket):
    """Envia heartbeat para manter a conexão WebSocket ativa."""
    try:
        while True:
            await asyncio.sleep(30)
            try:
                if websocket.client_state.CONNECTED:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                    logger.debug("WebSocket: Heartbeat enviado")
            except WebSocketDisconnect:
                logger.info("WebSocket: Cliente desconectado durante heartbeat")
                break
            except Exception as e:
                logger.error(f"WebSocket: Erro ao enviar heartbeat: {str(e)}")
                break
    except asyncio.CancelledError:
        logger.info("WebSocket: Heartbeat cancelado")
        pass
    except Exception as e:
        logger.error(f"WebSocket: Erro no heartbeat: {str(e)}") 
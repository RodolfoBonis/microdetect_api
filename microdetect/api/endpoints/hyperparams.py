from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging

from microdetect.database.database import get_db
from microdetect.models.hyperparam_search import HyperparamSearchStatus
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
        search = await hyperparam_service.create_search(search_data, db)
        
        # Iniciar busca em background usando o novo método que utiliza ProcessPoolExecutor
        background_tasks.add_task(
            hyperparam_service.start_search,
            search.id,
            db
        )
        
        # Converter para esquema de resposta
        response = HyperparamSearchResponse.from_orm(search)
        return build_response(response)
    except Exception as e:
        return build_error_response(str(e), 400)

@router.get("/", response_model=None)
async def list_hyperparam_searches(
    dataset_id: Optional[int] = None,
    status: Optional[HyperparamSearchStatus] = None,
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
    await websocket.accept()
    
    try:
        # Verificar se a busca existe
        search = await hyperparam_service.get_search(search_id, db)
        if not search:
            error_json = json.dumps({"error": "Busca não encontrada"}, cls=JSONEncoder)
            await websocket.send_text(error_json)
            await websocket.close()
            return
        
        # Configurar heartbeat
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        
        # Obter dados de progresso em tempo real
        progress_data = hyperparam_service.get_progress(search_id)
        
        # Enviar estado inicial
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
        
        # Monitorar atualizações
        last_update = None
        last_trials = None
        last_best_params = None
        last_best_metrics = None
        
        while True:
            try:
                # Obter dados de progresso atualizados
                progress_data = hyperparam_service.get_progress(search_id)
                
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
                    
                    # Atualizar busca para ter os dados mais recentes
                    db.refresh(search)
                    response = HyperparamSearchResponse.from_orm(search)
                    update_data = response.dict()
                    
                    # Adaptar os dados para o frontend
                    update_data.update({
                        "trials_data": current_trials,
                        "current_iteration": progress_data.get("current_iteration", 0),
                        "iterations_completed": progress_data.get("iterations_completed", 0),
                        "best_params": current_best_params,
                        "best_metrics": current_best_metrics,
                        "progress": progress_data
                    })

                    json_data = json.dumps(update_data, cls=JSONEncoder)
                    await websocket.send_text(json_data)
                
                # Verificar se a busca terminou
                if progress_data.get("status") in ["completed", "failed"]:
                    # Enviar uma atualização final
                    db.refresh(search)
                    response = HyperparamSearchResponse.from_orm(search)
                    final_data = response.dict()
                    final_data.update({
                        "trials_data": current_trials,
                        "best_params": current_best_params,
                        "best_metrics": current_best_metrics,
                        "status": progress_data.get("status", search.status),
                        "progress": progress_data
                    })
                    json_data = json.dumps(final_data, cls=JSONEncoder)
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
        logger.info(f"Cliente desconectou do websocket de hiperparâmetros {search_id}")
    except Exception as e:
        logger.error(f"Erro no websocket de hiperparâmetros: {str(e)}")
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
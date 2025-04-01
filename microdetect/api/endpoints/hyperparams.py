from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging

from microdetect.database.database import get_db
from microdetect.models.hyperparam_search import HyperparamSearch, HyperparamSearchStatus
from microdetect.schemas.hyperparam_search import (
    HyperparamSearchCreate,
    HyperparamSearchResponse,
    HyperparamSearchUpdate
)
from microdetect.services.hyperparam_service import HyperparamService
from microdetect.utils.serializers import build_response, build_error_response

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
            await websocket.send_json({"error": "Busca não encontrada"})
            await websocket.close()
            return
        
        # Obter dados de progresso em tempo real
        progress_data = hyperparam_service.get_progress(search_id)
        
        # Enviar estado inicial
        response = HyperparamSearchResponse.from_orm(search)
        initial_data = response.dict()
        
        # Adaptar os dados para o frontend
        initial_data.update({
            # Campos esperados pelo frontend
            "trials_data": progress_data.get("trials", []),
            "current_iteration": progress_data.get("current_iteration", 0),
            "iterations_completed": progress_data.get("iterations_completed", 0),
            # Também incluir dados de progresso raw para debug
            "progress": progress_data,
        })
        
        await websocket.send_json(initial_data)
        
        # Monitorar atualizações
        last_update = None
        while True:
            # Obter dados de progresso atualizados
            progress_data = hyperparam_service.get_progress(search_id)
            
            # Se os dados são diferentes da última atualização, enviar
            if last_update != progress_data:
                last_update = progress_data.copy()
                
                # Atualizar busca para ter os dados mais recentes
                db.refresh(search)
                response = HyperparamSearchResponse.from_orm(search)
                update_data = response.dict()
                
                # Adaptar os dados para o frontend
                update_data.update({
                    # Campos esperados pelo frontend
                    "trials_data": progress_data.get("trials", []),
                    "current_iteration": progress_data.get("current_iteration", 0),
                    "iterations_completed": progress_data.get("iterations_completed", 0),
                    # Campos para compatibilidade
                    "best_params": progress_data.get("best_params", {}),
                    "best_metrics": progress_data.get("best_metrics", {}),
                    # Incluir dados de progresso raw para debug
                    "progress": progress_data,
                })
                
                # Enviar atualização
                await websocket.send_json(update_data)
                
            # Verificar se a busca terminou baseado nos dados de progresso
            if progress_data.get("status") in ["completed", "failed"]:
                # Enviar uma atualização final
                db.refresh(search)
                response = HyperparamSearchResponse.from_orm(search)
                final_data = response.dict()
                final_data.update({
                    "trials_data": progress_data.get("trials", []),
                    "best_params": progress_data.get("best_params", {}),
                    "best_metrics": progress_data.get("best_metrics", {}),
                    "status": progress_data.get("status", search.status),
                    "progress": progress_data
                })
                await websocket.send_json(final_data)
                break
            
            # Aguardar próxima atualização
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        # Cliente desconectou
        pass
    except Exception as e:
        # Erro durante monitoramento
        logger.error(f"Erro no websocket: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Garantir que o websocket seja fechado
        try:
            await websocket.close()
        except:
            pass 
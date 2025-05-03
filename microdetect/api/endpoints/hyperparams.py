from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging
import uuid

from microdetect.database.database import get_db
from microdetect.models.training_session import TrainingStatus
from microdetect.models.hyperparam_search import HyperparamSearch
from microdetect.schemas.hyperparam_search import (
    HyperparamSearchResponse,
)
from microdetect.services.hyperparam_service import HyperparamService
from microdetect.utils.serializers import build_response, build_error_response, serialize_to_dict, JSONEncoder
from microdetect.core.websocket_utils import send_hyperparam_update, WS_CHANNEL_PREFIX, websocket_manager

router = APIRouter()
hyperparam_service = HyperparamService()
logger = logging.getLogger(__name__)

# Não precisamos inicializar novamente o gerenciador de WebSockets, usamos a instância de websocket_utils

# Constantes
# Removido WS_CHANNEL_PREFIX, importado do websocket_utils

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
            search_space=search_data.get("search_space")
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
async def websocket_endpoint(websocket: WebSocket, search_id: str, db: Session = Depends(get_db)):
    """Endpoint WebSocket para monitoramento de buscas de hiperparâmetros"""
    client_id = str(uuid.uuid4())
    
    # Conectar cliente (sem verificação de token por enquanto)
    await websocket_manager.connect(client_id, websocket)
    
    try:
        # Verificar se a busca de hiperparâmetros existe
        search = db.query(HyperparamSearch).filter(HyperparamSearch.id == search_id).first()
        if not search:
            await websocket.send_json({
                "error": "Busca de hiperparâmetros não encontrada",
                "code": 4004
            })
            await websocket_manager.disconnect(client_id)
            return
        
        # Registrar cliente para receber atualizações da busca
        websocket_manager.register_client_for_session(client_id, search_id)
        logger.info(f"Cliente {client_id} registrado para receber atualizações da busca {search_id}")
        
        # Enviar estado inicial
        search_state = {
            "id": search.id,
            "status": search.status,
            "created_at": search.created_at.isoformat() if search.created_at else None,
            "updated_at": search.updated_at.isoformat() if search.updated_at else None,
            "dataset_id": search.dataset_id,
            "best_trial": getattr(search, "best_trial", None),
            "search_space": search.search_space,
            "current_trial": getattr(search, "current_trial", None),
            "progress": getattr(search, "progress", None),
            "error": getattr(search, "error", None)
        }
        
        await websocket.send_json({
            "type": "initial_state",
            "data": search_state
        })
        
        # Esperar confirmação do cliente
        try:
            data = await websocket.receive_json()
            if data.get("type") == "acknowledge":
                logger.info(f"Cliente {client_id} confirmou o recebimento do estado inicial")
        except Exception as e:
            logger.warning(f"Erro ao receber confirmação: {e}")
        
        # Função de heartbeat
        async def send_heartbeat():
            while True:
                try:
                    if not websocket_manager.active_connections.get(client_id):
                        break
                    await websocket.send_json({"type": "heartbeat", "time": str(datetime.now())})
                    await asyncio.sleep(30)  # Enviar heartbeat a cada 30 segundos
                except Exception as e:
                    logger.error(f"Erro no heartbeat: {e}")
                    break
        
        # Iniciar heartbeat em segundo plano
        heartbeat_task = asyncio.create_task(send_heartbeat())
        
        # Loop principal para receber mensagens
        try:
            while True:
                data = await websocket.receive_json()
                if "type" in data:
                    if data["type"] == "acknowledge":
                        logger.info(f"Cliente {client_id} confirmou: {data.get('message', '')}")
                    elif data["type"] == "close":
                        logger.info(f"Cliente {client_id} solicitou fechamento da conexão")
                        break
        except WebSocketDisconnect:
            logger.info(f"Cliente {client_id} desconectado")
        finally:
            # Cancelar task de heartbeat
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            
            # Desconectar cliente
            await websocket_manager.disconnect(client_id)
    
    except Exception as e:
        logger.error(f"Erro no endpoint WebSocket: {e}")
        # Garantir que o cliente seja desconectado em caso de erro
        await websocket_manager.disconnect(client_id)
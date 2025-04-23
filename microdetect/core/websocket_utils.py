import asyncio
import logging
from microdetect.core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Inicializar o gerenciador de WebSockets (agora é um singleton)
websocket_manager = WebSocketManager()

# Constantes
WS_CHANNEL_PREFIX = "hyperparams"

async def send_hyperparam_update(search_id: str, data: dict):
    """
    Envia uma atualização sobre uma busca de hiperparâmetros para todos os clientes conectados.
    
    Args:
        search_id: ID da busca de hiperparâmetros
        data: Dados a serem enviados (serão convertidos para JSON)
    """
    channel = f"{WS_CHANNEL_PREFIX}_{search_id}"
    logger.debug(f"Enviando mensagem WebSocket para canal {channel}: {data}")
    logger.info(f"Enviando atualização para clientes conectados à busca ID {search_id}")
    await websocket_manager.broadcast_json(channel, data)
    logger.debug(f"Mensagem enviada com sucesso para o canal {channel}") 
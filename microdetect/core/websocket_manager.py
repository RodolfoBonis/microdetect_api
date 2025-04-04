import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Conecta um cliente WebSocket.
        
        Args:
            websocket: Conexão WebSocket
            client_id: ID do cliente (ex: "training_1", "hyperparam_1")
        """
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        logger.info(f"Cliente WebSocket conectado: {client_id}")
    
    async def disconnect(self, websocket: WebSocket, client_id: str):
        """
        Desconecta um cliente WebSocket.
        
        Args:
            websocket: Conexão WebSocket
            client_id: ID do cliente
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        logger.info(f"Cliente WebSocket desconectado: {client_id}")
    
    async def broadcast_json(self, client_id: str, data: Dict[str, Any]):
        """
        Envia dados JSON para todos os clientes conectados com o ID especificado.
        
        Args:
            client_id: ID do cliente
            data: Dados a serem enviados
        """
        if client_id not in self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections[client_id]:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Erro ao enviar dados para cliente {client_id}: {e}")
                disconnected.add(connection)
        
        # Remover conexões desconectadas
        for connection in disconnected:
            await self.disconnect(connection, client_id)
    
    async def broadcast_text(self, client_id: str, message: str):
        """
        Envia uma mensagem de texto para todos os clientes conectados com o ID especificado.
        
        Args:
            client_id: ID do cliente
            message: Mensagem a ser enviada
        """
        if client_id not in self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections[client_id]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para cliente {client_id}: {e}")
                disconnected.add(connection)
        
        # Remover conexões desconectadas
        for connection in disconnected:
            await self.disconnect(connection, client_id) 
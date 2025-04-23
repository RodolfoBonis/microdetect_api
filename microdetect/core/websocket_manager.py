import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, HTTPException
from fastapi.security import HTTPBearer

logger = logging.getLogger(__name__)

security = HTTPBearer()

# Instância global singleton
_instance = None

class WebSocketManager:
    """Gerenciador de conexões WebSocket"""

    def __new__(cls):
        global _instance
        if _instance is None:
            _instance = super(WebSocketManager, cls).__new__(cls)
            _instance.active_connections = {}
            _instance.session_clients = {}
            _instance.client_sessions = {}
            _instance.logger = logging.getLogger(__name__)
            logger.info("Nova instância de WebSocketManager criada (singleton)")
        return _instance

    def __init__(self):
        # Inicialização já feita em __new__, este método não faz nada
        pass

    def verify_token(self, token: str) -> bool:
        """Verifica se o token é válido - sempre retorna True para desabilitar autenticação"""
        # Apenas retorna True para qualquer token (ou sem token)
        return True

    async def connect(self, client_id: str, websocket: WebSocket):
        """Conecta um novo cliente WebSocket"""
        # Aceitar sem verificar token
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.logger.info(f"Cliente {client_id} conectado")
        except Exception as e:
            self.logger.error(f"Erro ao conectar cliente {client_id}: {e}")
            raise

    async def disconnect(self, client_id: str):
        """Desconecta um cliente WebSocket"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            # Desregistrar cliente de todas as sessões
            if client_id in self.client_sessions:
                # Fazer uma cópia do conjunto para evitar erro de modificação durante iteração
                session_ids = set(self.client_sessions[client_id])
                for session_id in session_ids:
                    self.unregister_client_from_session(client_id, session_id)
                
            # Remover cliente da lista de conexões ativas
            del self.active_connections[client_id]
            
            # Fechar websocket se ainda estiver conectado
            try:
                await websocket.close()
            except Exception as e:
                self.logger.error(f"Erro ao fechar websocket para cliente {client_id}: {e}")
                
            self.logger.info(f"Cliente {client_id} desconectado")

    async def send_personal_message(self, client_id: str, message: str):
        """Envia uma mensagem de texto para um cliente específico"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
                return True
            except Exception as e:
                self.logger.error(f"Erro ao enviar mensagem para cliente {client_id}: {e}")
                return False
        return False

    async def broadcast_to_session(self, session_id: str, message: str):
        """Envia uma mensagem para todos os clientes de uma sessão"""
        if session_id not in self.session_clients:
            self.logger.warning(f"Tentativa de enviar mensagem para sessão inexistente: {session_id}")
            return
        
        self.logger.info(f"Enviando mensagem para sessão {session_id} com {len(self.session_clients[session_id])} clientes")
            
        disconnected_clients = set()
        for client_id in self.session_clients[session_id]:
            success = await self.send_personal_message(client_id, message)
            if success:
                self.logger.debug(f"Mensagem enviada com sucesso para cliente {client_id}")
            else:
                self.logger.warning(f"Falha ao enviar mensagem para cliente {client_id}")
                disconnected_clients.add(client_id)
        
        # Remover clientes desconectados
        for client_id in disconnected_clients:
            self.unregister_client_from_session(client_id, session_id)

    def register_client_for_session(self, client_id: str, session_id: str):
        """Registra um cliente para receber atualizações de uma sessão"""
        # Adicionar sessão à lista de sessões do cliente
        if client_id not in self.client_sessions:
            self.client_sessions[client_id] = set()
        self.client_sessions[client_id].add(session_id)
        
        # Adicionar cliente à lista de clientes da sessão
        if session_id not in self.session_clients:
            self.session_clients[session_id] = set()
        self.session_clients[session_id].add(client_id)
        
        self.logger.info(f"Cliente {client_id} registrado para sessão {session_id}")

    def unregister_client_from_session(self, client_id: str, session_id: str):
        """Remove um cliente da lista de clientes de uma sessão"""
        # Remover sessão da lista de sessões do cliente
        if client_id in self.client_sessions and session_id in self.client_sessions[client_id]:
            self.client_sessions[client_id].remove(session_id)
            if not self.client_sessions[client_id]:
                del self.client_sessions[client_id]
        
        # Remover cliente da lista de clientes da sessão
        if session_id in self.session_clients and client_id in self.session_clients[session_id]:
            self.session_clients[session_id].remove(client_id)
            if not self.session_clients[session_id]:
                del self.session_clients[session_id]
            
        self.logger.info(f"Cliente {client_id} desregistrado da sessão {session_id}")

    def get_session_clients(self, session_id: str) -> Set[str]:
        """Retorna a lista de clientes conectados a uma sessão"""
        return self.session_clients.get(session_id, set())

    def get_client_sessions(self, client_id: str) -> Set[str]:
        """Retorna a lista de sessões de um cliente"""
        return self.client_sessions.get(client_id, set())

    async def broadcast_json(self, channel: str, data: dict):
        """
        Envia uma mensagem JSON para todos os clientes registrados em um canal.
        
        Args:
            channel: Nome do canal (geralmente no formato 'tipo_id')
            data: Dados a serem enviados como JSON
        """
        self.logger.info(f"Tentando enviar mensagem JSON para canal: {channel}")
        
        # Extrair ID da sessão do canal (formato: tipo_id)
        session_id = None
        if "_" in channel:
            try:
                # Extrair a parte do ID (após o primeiro _)
                session_id = channel.split("_", 1)[1]
                self.logger.info(f"ID extraído do canal {channel}: {session_id}")
            except Exception as e:
                self.logger.error(f"Erro ao extrair ID da sessão do canal {channel}: {e}")
        else:
            # Se não tiver prefixo, usar o próprio canal como ID
            session_id = channel
            self.logger.info(f"Usando canal como ID de sessão: {session_id}")
            
        if not session_id:
            self.logger.error(f"Não foi possível extrair ID de sessão do canal: {channel}")
            return
        
        # Usar o ID exato do canal para a sessão
        if session_id not in self.session_clients:
            # Tentar remover prefixos comuns se não encontrar a sessão
            fallback_ids = [session_id]
            # Não modificar esta parte - apenas para debug
            self.logger.debug(f"Testando sessões disponíveis: {list(self.session_clients.keys())}")
            
        try:
            # Garantir que os dados são serializáveis para JSON
            # Converter exceções em strings
            for key, value in data.items():
                if isinstance(value, Exception) or isinstance(value, NotImplementedError):
                    data[key] = str(value)
            
            # Verificar quantos clientes estão registrados
            client_count = len(self.session_clients.get(session_id, []))
            self.logger.info(f"Enviando dados JSON para sessão {session_id} ({client_count} clientes)")
            
            # Enviar a mensagem
            await self.broadcast_to_session(session_id, json.dumps(data))
            
        except TypeError as e:
            # Caso ainda ocorra erro de serialização, tentar serializar um objeto simplificado
            self.logger.error(f"Erro ao serializar dados para JSON: {e}")
            await self.broadcast_to_session(session_id, json.dumps({
                "error": "Erro ao serializar dados",
                "type": "error"
            }))

    async def receive_json(self, client_id: str) -> Any:
        """Recebe uma mensagem JSON de um cliente específico"""
        if client_id in self.active_connections:
            return await self.active_connections[client_id].receive_json()
        return None

    async def is_connected(self, client_id: str) -> bool:
        """Verifica se um cliente está conectado"""
        return client_id in self.active_connections

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
            await self.disconnect(client_id) 
# Melhorias nos WebSockets

Este documento descreve as melhorias implementadas nos WebSockets para monitoramento em tempo real de treinamento e busca de hiperparâmetros.

## Visão Geral

As melhorias focam em quatro aspectos principais:

1. **Confiabilidade da Conexão**
   - Sistema de heartbeat
   - Reconexão automática
   - Tratamento robusto de erros

2. **Performance**
   - Atualizações mais frequentes
   - Otimização de dados
   - Gerenciamento eficiente de recursos

3. **Experiência do Usuário**
   - Feedback em tempo real
   - Indicadores de status
   - Tratamento gracioso de falhas

4. **Manutenibilidade**
   - Código modular
   - Logs detalhados
   - Documentação clara

## Sistema de Heartbeat

### Implementação

- Heartbeats são enviados a cada 30 segundos
- O cliente deve responder ou reconectar
- Conexão é fechada após 60 segundos sem heartbeat

### Exemplo de Código

```python
async def send_heartbeat(websocket: WebSocket):
    while True:
        try:
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Erro ao enviar heartbeat: {e}")
            break
```

## Reconexão Automática

### Estratégia

1. Detectar desconexão
2. Esperar 5 segundos
3. Tentar reconectar
4. Repetir até sucesso ou cancelamento

### Exemplo de Código

```python
async def reconnect_websocket(session_id: int):
    while True:
        try:
            await asyncio.sleep(5)
            logger.info(f"Tentando reconectar WebSocket para sessão {session_id}")
            await connect_websocket(session_id)
            break
        except Exception as e:
            logger.error(f"Erro na reconexão: {e}")
```

## Otimização de Dados

### Estratégias

1. **Verificação de Mudanças**
   - Enviar atualizações apenas quando houver mudanças significativas
   - Comparar valores anteriores com atuais

2. **Limitação de Histórico**
   - Manter apenas os últimos 100 valores para gráficos
   - Limpar dados antigos automaticamente

3. **Compressão de Dados**
   - Formato JSON otimizado
   - Remoção de campos redundantes

### Exemplo de Código

```python
def should_send_update(current: dict, previous: dict) -> bool:
    # Verificar mudanças significativas
    if abs(current.get('loss', 0) - previous.get('loss', 0)) > 0.001:
        return True
    
    if abs(current.get('map50', 0) - previous.get('map50', 0)) > 0.001:
        return True
    
    # Verificar mudanças em recursos
    resources_current = current.get('resources', {})
    resources_previous = previous.get('resources', {})
    
    if abs(resources_current.get('cpu_percent', 0) - 
           resources_previous.get('cpu_percent', 0)) > 5:
        return True
    
    return False
```

## Tratamento de Erros

### Estratégias

1. **Logging Detalhado**
   - Registrar todos os eventos importantes
   - Incluir contexto nos logs
   - Níveis apropriados de log

2. **Recuperação Graciosa**
   - Tentar recuperar de erros comuns
   - Limpar recursos adequadamente
   - Notificar o usuário quando necessário

3. **Monitoramento**
   - Rastrear métricas de conexão
   - Alertas para problemas recorrentes
   - Estatísticas de uso

### Exemplo de Código

```python
async def handle_websocket_error(websocket: WebSocket, error: Exception):
    logger.error(f"Erro no WebSocket: {error}")
    
    try:
        # Tentar enviar mensagem de erro para o cliente
        await websocket.send_json({
            "type": "error",
            "message": str(error)
        })
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem de erro: {e}")
    
    finally:
        # Limpar recursos
        await cleanup_resources(websocket)
```

## Integração com o Frontend

### Exemplo de Código Flutter

```dart
class WebSocketManager {
  WebSocketChannel? _channel;
  Timer? _heartbeatTimer;
  DateTime? _lastHeartbeat;
  
  Future<void> connect(String url) async {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(url));
      _startHeartbeatMonitoring();
      
      _channel!.stream.listen(
        (message) {
          _handleMessage(message);
        },
        onDone: () {
          _handleDisconnect();
        },
        onError: (error) {
          _handleError(error);
        },
      );
    } catch (e) {
      _handleError(e);
    }
  }
  
  void _startHeartbeatMonitoring() {
    _lastHeartbeat = DateTime.now();
    _heartbeatTimer?.cancel();
    
    _heartbeatTimer = Timer.periodic(
      const Duration(seconds: 30),
      (timer) {
        final now = DateTime.now();
        if (_lastHeartbeat != null &&
            now.difference(_lastHeartbeat!).inSeconds > 60) {
          _handleDisconnect();
          timer.cancel();
        }
      },
    );
  }
  
  void _handleMessage(dynamic message) {
    try {
      final data = jsonDecode(message as String);
      
      if (data['type'] == 'heartbeat') {
        _lastHeartbeat = DateTime.now();
        return;
      }
      
      // Processar mensagem
      _processMessage(data);
    } catch (e) {
      _handleError(e);
    }
  }
  
  void _handleDisconnect() {
    _channel?.sink.close();
    _channel = null;
    _heartbeatTimer?.cancel();
    
    // Tentar reconectar
    Future.delayed(
      const Duration(seconds: 5),
      () => connect(_url),
    );
  }
  
  void _handleError(dynamic error) {
    logger.error('Erro no WebSocket: $error');
    _handleDisconnect();
  }
}
```

## Métricas e Monitoramento

### Métricas Importantes

1. **Conexão**
   - Taxa de reconexão
   - Tempo médio entre falhas
   - Latência de mensagens

2. **Performance**
   - Uso de CPU/memória
   - Taxa de mensagens
   - Tamanho médio das mensagens

3. **Qualidade**
   - Taxa de erro
   - Tempo de resposta
   - Qualidade da conexão

### Exemplo de Código

```python
class WebSocketMetrics:
    def __init__(self):
        self.connection_attempts = 0
        self.successful_connections = 0
        self.failed_connections = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.errors = 0
        self.start_time = datetime.now()
    
    def log_connection_attempt(self, success: bool):
        self.connection_attempts += 1
        if success:
            self.successful_connections += 1
        else:
            self.failed_connections += 1
    
    def log_message(self, sent: bool):
        if sent:
            self.messages_sent += 1
        else:
            self.messages_received += 1
    
    def log_error(self):
        self.errors += 1
    
    def get_stats(self) -> dict:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "uptime_seconds": uptime,
            "connection_success_rate": (
                self.successful_connections / self.connection_attempts
                if self.connection_attempts > 0 else 0
            ),
            "messages_per_second": (
                (self.messages_sent + self.messages_received) / uptime
                if uptime > 0 else 0
            ),
            "error_rate": (
                self.errors / (self.messages_sent + self.messages_received)
                if (self.messages_sent + self.messages_received) > 0 else 0
            )
        }
```

## Próximos Passos

1. **Melhorias Futuras**
   - Compressão de dados
   - Cache de mensagens
   - Balanceamento de carga

2. **Monitoramento**
   - Dashboard de métricas
   - Alertas automáticos
   - Análise de padrões

3. **Documentação**
   - Guias de implementação
   - Exemplos de uso
   - Troubleshooting 
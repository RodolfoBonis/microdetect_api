# Documentação das Mensagens WebSocket

## Visão Geral

O sistema utiliza WebSocket para comunicação em tempo real entre o servidor e os clientes. Esta documentação descreve os tipos de mensagens, seus formatos e exemplos de uso.

## Tipos de Mensagens

### 1. Estado Inicial (initial_state)

Enviado quando um cliente se conecta ao WebSocket para receber o estado atual da sessão.

**Formato:**
```json
{
    "type": "initial_state",
    "data": {
        "id": "string",
        "status": "string",
        "created_at": "string (ISO format)",
        "updated_at": "string (ISO format)",
        "metadata": {},
        "error": "string | null",
        "result": {},
        "progress": {}
    }
}
```

**Exemplo de Dados:**
```json
{
    "type": "initial_state",
    "data": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "pending",
        "created_at": "2024-03-20T15:30:00.000Z",
        "updated_at": "2024-03-20T15:30:00.000Z",
        "metadata": {
            "dataset_id": 1,
            "model_type": "yolov8",
            "model_version": "n"
        },
        "error": null,
        "result": null,
        "progress": {
            "current_epoch": 0,
            "total_epochs": 100
        }
    }
}
```

### 2. Heartbeat

Mensagem enviada periodicamente para manter a conexão ativa.

**Formato:**
```json
{
    "type": "heartbeat",
    "time": "string (ISO format)"
}
```

**Exemplo de Dados:**
```json
{
    "type": "heartbeat",
    "time": "2024-03-20T15:30:30.000Z"
}
```

### 3. Atualização de Treinamento

Enviado durante o processo de treinamento para atualizar o progresso.

**Formato:**
```json
{
    "status": "training",
    "metrics": {
        "map50": float,
        "map50_95": float,
        "precision": float,
        "recall": float,
        "fitness": float
    },
    "current_epoch": int,
    "total_epochs": int,
    "progress": {
        "current_epoch": int,
        "total_epochs": int,
        "percent_complete": float,
        "progress_type": "string"
    }
}
```

**Exemplo de Dados:**
```json
{
    "status": "training",
    "metrics": {
        "map50": 0.85,
        "map50_95": 0.65,
        "precision": 0.92,
        "recall": 0.88,
        "fitness": 0.90
    },
    "current_epoch": 25,
    "total_epochs": 100,
    "progress": {
        "current_epoch": 25,
        "total_epochs": 100,
        "percent_complete": 25.0,
        "progress_type": "epoch"
    }
}
```

### 4. Conclusão de Treinamento

Enviado quando o treinamento é concluído com sucesso.

**Formato:**
```json
{
    "status": "completed",
    "metrics": {
        "map50": float,
        "map50_95": float,
        "precision": float,
        "recall": float,
        "fitness": float
    },
    "message": "Treinamento concluído com sucesso"
}
```

**Exemplo de Dados:**
```json
{
    "status": "completed",
    "metrics": {
        "map50": 0.92,
        "map50_95": 0.75,
        "precision": 0.95,
        "recall": 0.93,
        "fitness": 0.94
    },
    "message": "Treinamento concluído com sucesso"
}
```

### 5. Erro de Treinamento

Enviado quando ocorre um erro durante o treinamento.

**Formato:**
```json
{
    "status": "failed",
    "error": "string",
    "message": "Erro durante o treinamento"
}
```

**Exemplo de Dados:**
```json
{
    "status": "failed",
    "error": "Erro ao carregar dataset: Arquivo não encontrado",
    "message": "Erro durante o treinamento"
}
```

### 6. Confirmação do Cliente (acknowledge)

Enviado pelo cliente para confirmar o recebimento de mensagens.

**Formato:**
```json
{
    "type": "acknowledge",
    "message": "string (opcional)"
}
```

**Exemplo de Dados:**
```json
{
    "type": "acknowledge",
    "message": "Estado inicial recebido"
}
```

### 7. Solicitação de Fechamento (close)

Enviado pelo cliente para solicitar o fechamento da conexão.

**Formato:**
```json
{
    "type": "close"
}
```

### 8. Atualização de Busca de Hiperparâmetros

Enviado durante o processo de busca de hiperparâmetros para atualizar o progresso.

**Formato:**
```json
{
    "status": "searching",
    "trials": [
        {
            "trial_id": "string",
            "params": {
                "learning_rate": float,
                "batch_size": int,
                "epochs": int,
                // outros parâmetros específicos
            },
            "metrics": {
                "map50": float,
                "map50_95": float,
                "precision": float,
                "recall": float,
                "fitness": float
            },
            "status": "string"
        }
    ],
    "best_params": {
        "learning_rate": float,
        "batch_size": int,
        "epochs": int,
        // outros parâmetros específicos
    },
    "best_metrics": {
        "map50": float,
        "map50_95": float,
        "precision": float,
        "recall": float,
        "fitness": float
    },
    "current_trial": int,
    "total_trials": int,
    "progress": {
        "trial": int,
        "total_trials": int,
        "current_epoch": int,
        "total_epochs": int,
        "percent_complete": int
    },
    "current_trial_info": {
        "epoch": int,
        "progress_type": "string",
        "current_trial": int
    }
}
```

**Exemplo de Dados:**
```json
{
    "status": "searching",
    "trials": [
        {
            "trial_id": "trial_1",
            "params": {
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100
            },
            "metrics": {
                "map50": 0.85,
                "map50_95": 0.65,
                "precision": 0.92,
                "recall": 0.88,
                "fitness": 0.90
            },
            "status": "completed"
        }
    ],
    "best_params": {
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 100
    },
    "best_metrics": {
        "map50": 0.85,
        "map50_95": 0.65,
        "precision": 0.92,
        "recall": 0.88,
        "fitness": 0.90
    },
    "current_trial": 1,
    "total_trials": 10,
    "progress": {
        "trial": 1,
        "total_trials": 10,
        "current_epoch": 25,
        "total_epochs": 100,
        "percent_complete": 10
    },
    "current_trial_info": {
        "epoch": 25,
        "progress_type": "epoch_in_trial",
        "current_trial": 1
    }
}
```

### 9. Conclusão de Busca de Hiperparâmetros

Enviado quando a busca de hiperparâmetros é concluída com sucesso.

**Formato:**
```json
{
    "status": "completed",
    "best_params": {
        "learning_rate": float,
        "batch_size": int,
        "epochs": int,
        // outros parâmetros específicos
    },
    "best_metrics": {
        "map50": float,
        "map50_95": float,
        "precision": float,
        "recall": float,
        "fitness": float
    },
    "message": "Busca concluída com sucesso"
}
```

**Exemplo de Dados:**
```json
{
    "status": "completed",
    "best_params": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 150
    },
    "best_metrics": {
        "map50": 0.92,
        "map50_95": 0.75,
        "precision": 0.95,
        "recall": 0.93,
        "fitness": 0.94
    },
    "message": "Busca concluída com sucesso"
}
```

### 10. Erro na Busca de Hiperparâmetros

Enviado quando ocorre um erro durante a busca de hiperparâmetros.

**Formato:**
```json
{
    "status": "failed",
    "error": "string",
    "message": "Erro durante a busca"
}
```

**Exemplo de Dados:**
```json
{
    "status": "failed",
    "error": "Erro ao executar trial: Memória insuficiente",
    "message": "Erro durante a busca"
}
```

## Exemplos de Uso em Flutter

### Exemplo 1: Monitoramento de Treinamento

```dart
import 'package:web_socket_channel/web_socket_channel.dart';

class TrainingWebSocket {
  final String sessionId;
  late WebSocketChannel _channel;
  
  TrainingWebSocket(this.sessionId) {
    _channel = WebSocketChannel.connect(
      Uri.parse('ws://seu-servidor/training/ws/$sessionId'),
    );
  }
  
  void listen(void Function(Map<String, dynamic>) onMessage) {
    _channel.stream.listen(
      (dynamic message) {
        final data = jsonDecode(message);
        
        if (data['type'] == 'initial_state') {
          print('Estado inicial: ${data['data']}');
          // Enviar confirmação
          _channel.sink.add(jsonEncode({
            'type': 'acknowledge',
            'message': 'Estado inicial recebido'
          }));
        } else if (data['status'] == 'training') {
          print('Progresso: ${data['progress']['percent_complete']}%');
          print('Métricas: ${data['metrics']}');
        } else if (data['status'] == 'completed') {
          print('Treinamento concluído: ${data['metrics']}');
        } else if (data['status'] == 'failed') {
          print('Erro: ${data['error']}');
        }
        
        onMessage(data);
      },
      onError: (error) => print('Erro: $error'),
      onDone: () => print('Conexão fechada'),
    );
  }
  
  void close() {
    _channel.sink.close();
  }
}
```

### Exemplo 2: Monitoramento de Busca de Hiperparâmetros

```dart
import 'package:web_socket_channel/web_socket_channel.dart';

class HyperparamWebSocket {
  final String searchId;
  late WebSocketChannel _channel;
  
  HyperparamWebSocket(this.searchId) {
    _channel = WebSocketChannel.connect(
      Uri.parse('ws://seu-servidor/hyperparams/ws/$searchId'),
    );
  }
  
  void listen(void Function(Map<String, dynamic>) onMessage) {
    _channel.stream.listen(
      (dynamic message) {
        final data = jsonDecode(message);
        
        if (data['type'] == 'initial_state') {
          print('Estado inicial da busca: ${data['data']}');
        } else if (data['status'] == 'searching') {
          print('Progresso da busca: ${data['progress']['percent_complete']}%');
          print('Trial atual: ${data['current_trial']}/${data['total_trials']}');
          print('Melhores parâmetros até agora: ${data['best_params']}');
          print('Métricas do melhor trial: ${data['best_metrics']}');
        } else if (data['status'] == 'completed') {
          print('Busca concluída!');
          print('Melhores parâmetros encontrados: ${data['best_params']}');
          print('Métricas finais: ${data['best_metrics']}');
        } else if (data['status'] == 'failed') {
          print('Erro na busca: ${data['error']}');
        }
        
        onMessage(data);
      },
      onError: (error) => print('Erro: $error'),
      onDone: () => print('Conexão fechada'),
    );
  }
  
  void close() {
    _channel.sink.close();
  }
}
```

## Observações Importantes

1. Todas as mensagens são enviadas no formato JSON
2. O servidor envia heartbeats a cada 30 segundos para manter a conexão ativa
3. Os clientes devem enviar confirmações (acknowledge) para mensagens importantes
4. Em caso de erro, a conexão é fechada automaticamente
5. Os clientes podem solicitar o fechamento da conexão enviando uma mensagem do tipo "close"
6. Para usar em Flutter, adicione a dependência `web_socket_channel` no seu `pubspec.yaml`:
```yaml
dependencies:
  web_socket_channel: ^2.4.0
``` 
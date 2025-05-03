# Documentação MicroDetect API

## 1. Visão Geral

A MicroDetect API é uma solução completa para detecção de microorganismos em imagens microscópicas. Esta documentação detalha todos os endpoints, esquemas, websockets e fluxos de comunicação para integração com um frontend desenvolvido em Flutter.

## 2. Base URL e Conexões

- **Base URL (REST API)**: `http://seu-servidor:porta/api/v1`
- **Base URL (WebSockets)**: `ws://seu-servidor:porta/api/v1`

## 3. Autenticação

Atualmente o sistema não implementa autenticação, mas está preparado para adicionar no futuro.

## 4. Endpoints REST API

### 4.1. Datasets

#### 4.1.1. Criar Dataset
- **Endpoint**: `POST /datasets/`
- **Corpo da requisição**:
```json
{
  "name": "Nome do Dataset",
  "description": "Descrição opcional",
  "classes": [
    {"id": 0, "name": "Microorganismo 1"},
    {"id": 1, "name": "Microorganismo 2"}
  ]
}
```
- **Resposta**: `DatasetResponse`

#### 4.1.2. Listar Datasets
- **Endpoint**: `GET /datasets/`
- **Parâmetros de Query**:
  - `skip`: Posição inicial (padrão: 0)
  - `limit`: Número máximo de registros (padrão: 100)
- **Resposta**: Lista de `DatasetResponse`

#### 4.1.3. Obter Dataset
- **Endpoint**: `GET /datasets/{dataset_id}`
- **Resposta**: `DatasetResponse`

### 4.2. Imagens

#### 4.2.1. Upload de Imagem
- **Endpoint**: `POST /images/`
- **Corpo da requisição**: Form-data com:
  - `file`: Arquivo de imagem (JPEG, PNG, TIFF)
  - `dataset_id`: ID do dataset (opcional)
  - `metadata`: Metadados em formato JSON (opcional)
  - `width`, `height`: Dimensões (opcional)
- **Resposta**: `ImageResponse`

#### 4.2.2. Listar Imagens
- **Endpoint**: `GET /images/`
- **Parâmetros de Query**:
  - `dataset_id`: Filtrar por dataset (opcional)
  - `skip`, `limit`: Paginação
- **Resposta**: Lista de `ImageResponse`

### 4.3. Treinamento

#### 4.3.1. Criar Sessão de Treinamento
- **Endpoint**: `POST /training/`
- **Corpo da requisição**:
```json
{
  "name": "YOLOv8n Microorganismos",
  "model_type": "yolov8",
  "model_version": "n",
  "dataset_id": 123,
  "description": "Modelo para detectar microorganismos",
  "hyperparameters": {
    "batch_size": 16,
    "imgsz": 640,
    "epochs": 100,
    "device": "auto",
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 50
  }
}
```
- **Resposta**: `TrainingSessionResponse`

#### 4.3.2. Listar Sessões de Treinamento
- **Endpoint**: `GET /training/`
- **Parâmetros de Query**:
  - `dataset_id`: Filtrar por dataset (opcional)
  - `status`: Filtrar por status (opcional)
  - `skip`, `limit`: Paginação
- **Resposta**: Lista de `TrainingSessionResponse`

#### 4.3.3. Obter Sessão de Treinamento
- **Endpoint**: `GET /training/{session_id}`
- **Resposta**: `TrainingSessionResponse`

#### 4.3.4. Obter Relatório de Treinamento
- **Endpoint**: `GET /training/{session_id}/report`
- **Resposta**: `TrainingReportResponse`

### 4.4. Busca de Hiperparâmetros

#### 4.4.1. Criar Busca de Hiperparâmetros
- **Endpoint**: `POST /hyperparams/`
- **Corpo da requisição**:
```json
{
  "name": "Busca YOLOv8",
  "dataset_id": 123,
  "model_type": "yolov8",
  "model_version": "n",
  "description": "Otimização de hiperparâmetros",
  "search_space": {
    "batch_size": {"min": 8, "max": 32},
    "lr0": {"min": 0.001, "max": 0.1},
    "optimizer": {"choices": ["SGD", "Adam", "AdamW"]}
  },
  "max_trials": 5
}
```
- **Resposta**: `HyperparamSearchResponse`

#### 4.4.2. Listar Buscas de Hiperparâmetros
- **Endpoint**: `GET /hyperparams/`
- **Parâmetros de Query**:
  - `dataset_id`: Filtrar por dataset (opcional)
  - `status`: Filtrar por status (opcional)
  - `skip`, `limit`: Paginação
- **Resposta**: Lista de `HyperparamSearchResponse`

#### 4.4.3. Obter Busca de Hiperparâmetros
- **Endpoint**: `GET /hyperparams/{search_id}`
- **Resposta**: `HyperparamSearchResponse`

### 4.5. Modelos

#### 4.5.1. Listar Modelos
- **Endpoint**: `GET /models/`
- **Parâmetros de Query**:
  - `training_session_id`: Filtrar por sessão (opcional)
  - `model_type`: Filtrar por tipo (opcional)
  - `skip`, `limit`: Paginação
- **Resposta**: Lista de `ModelResponse`

#### 4.5.2. Obter Modelo
- **Endpoint**: `GET /models/{model_id}`
- **Resposta**: `ModelResponse`

### 4.6. Inferência

#### 4.6.1. Realizar Inferência
- **Endpoint**: `POST /inference/`
- **Corpo da requisição**: Form-data com:
  - `file`: Arquivo de imagem
  - `model_id`: ID do modelo
  - `confidence_threshold`: Limiar de confiança (opcional, padrão: 0.5)
- **Resposta**: `InferenceResultResponse`

## 5. WebSockets

### 5.1. Monitoramento de Treinamento

#### 5.1.1. Conectar ao WebSocket de Treinamento
- **URL**: `ws://seu-servidor:porta/api/v1/training/ws/{session_id}`
- **Mensagens Recebidas**:
  - Estado Inicial:
    ```json
    {
      "type": "initial_state",
      "data": {
        "id": 123,
        "status": "training",
        "created_at": "2023-04-23T12:34:56",
        "updated_at": "2023-04-23T12:35:00",
        "metadata": {},
        "error": null,
        "result": null,
        "progress": null
      }
    }
    ```
  - Heartbeat:
    ```json
    {
      "type": "heartbeat",
      "time": "2023-04-23T12:36:00"
    }
    ```
  - Atualizações de Progresso:
    ```json
    {
      "status": "training",
      "metrics": {
        "epoch": 3,
        "loss": 0.234,
        "map50": 0.728,
        "map": 0.456,
        "precision": 0.81,
        "recall": 0.79,
        "val_loss": 0.345
      },
      "current_epoch": 3,
      "total_epochs": 30,
      "progress": {
        "current_epoch": 3,
        "total_epochs": 30,
        "percent_complete": 10,
        "progress_type": "epoch"
      }
    }
    ```
  - Conclusão:
    ```json
    {
      "status": "completed",
      "metrics": {
        "epochs": 30,
        "best_epoch": 25,
        "best_map50": 0.863,
        "best_map": 0.612,
        "final_map50": 0.859,
        "final_map": 0.608,
        "train_time": 1345.6,
        "val_time": 89.2
      },
      "message": "Treinamento concluído com sucesso"
    }
    ```
  - Erro:
    ```json
    {
      "status": "error",
      "error": "Mensagem de erro",
      "message": "Erro ao monitorar progresso"
    }
    ```

#### 5.1.2. Mensagens para Enviar ao WebSocket
- Confirmar recebimento:
  ```json
  {
    "type": "acknowledge",
    "message": "Estado inicial recebido"
  }
  ```
- Fechar conexão:
  ```json
  {
    "type": "close"
  }
  ```

### 5.2. Monitoramento de Busca de Hiperparâmetros

#### 5.2.1. Conectar ao WebSocket de Hiperparâmetros
- **URL**: `ws://seu-servidor:porta/api/v1/hyperparams/ws/{search_id}`
- **Mensagens Recebidas**:
  - Estado Inicial:
    ```json
    {
      "type": "initial_state",
      "data": {
        "id": 123,
        "status": "running",
        "created_at": "2023-04-23T12:34:56",
        "updated_at": "2023-04-23T12:35:00",
        "dataset_id": 456,
        "best_trial": null,
        "search_space": {},
        "current_trial": 0,
        "progress": null,
        "error": null
      }
    }
    ```
  - Heartbeat: (mesmo formato do treinamento)
  - Atualizações de Progresso:
    ```json
    {
      "status": "running",
      "trials": [
        {
          "trial": 1,
          "params": {"batch_size": 16, "lr0": 0.01},
          "metrics": {"map50": 0.756, "map": 0.512}
        }
      ],
      "best_params": {"batch_size": 16, "lr0": 0.01},
      "best_metrics": {"map50": 0.756, "map": 0.512},
      "current_trial": 1,
      "total_trials": 5,
      "progress": {
        "trial": 1,
        "total_trials": 5,
        "current_epoch": 3,
        "total_epochs": 30,
        "percent_complete": 10
      },
      "current_trial_info": {
        "epoch": 3,
        "total_epochs": 30,
        "loss": 0.234,
        "progress_type": "epoch_in_trial"
      }
    }
    ```
  - Conclusão:
    ```json
    {
      "status": "completed",
      "best_params": {"batch_size": 32, "lr0": 0.05},
      "best_metrics": {"map50": 0.842, "map": 0.623},
      "message": "Busca concluída com sucesso"
    }
    ```

## 6. Esquemas de Dados

### 6.1. DatasetResponse
```json
{
  "id": 123,
  "name": "Dataset Microorganismos",
  "description": "Dataset para treinamento",
  "classes": [
    {"id": 0, "name": "Microorganismo A"},
    {"id": 1, "name": "Microorganismo B"}
  ],
  "images_count": 500,
  "annotations_count": 1250,
  "created_at": "2023-04-20T10:15:30",
  "updated_at": "2023-04-20T10:15:30"
}
```

### 6.2. ImageResponse
```json
{
  "id": 456,
  "file_name": "imagem_microscopia.jpg",
  "file_path": "/path/to/image",
  "file_size": 1024000,
  "url": "/images/imagem_microscopia.jpg",
  "width": 1280,
  "height": 720,
  "image_metadata": {},
  "dataset_id": 123,
  "datasets": [
    {"id": 123, "name": "Dataset Microorganismos"}
  ],
  "annotations": [],
  "created_at": "2023-04-20T11:30:45",
  "updated_at": "2023-04-20T11:30:45"
}
```

### 6.3. TrainingSessionResponse
```json
{
  "id": 789,
  "name": "Treinamento YOLOv8n",
  "description": "Sessão de treinamento para detector",
  "model_type": "yolov8",
  "model_version": "n",
  "dataset_id": 123,
  "hyperparameters": {
    "batch_size": 16,
    "imgsz": 640,
    "epochs": 100
  },
  "status": "training",
  "metrics": {
    "current_epoch": 45,
    "loss": 0.234,
    "map50": 0.756
  },
  "created_at": "2023-04-21T09:00:00",
  "updated_at": "2023-04-21T09:10:30",
  "started_at": "2023-04-21T09:01:15",
  "completed_at": null
}
```

### 6.4. ModelResponse
```json
{
  "id": 321,
  "name": "Modelo YOLOv8n Microorganismos",
  "description": "Modelo treinado",
  "filepath": "/path/to/model.pt",
  "model_type": "yolov8",
  "model_version": "n",
  "metrics": {
    "map50": 0.812,
    "map": 0.653
  },
  "training_session_id": 789,
  "created_at": "2023-04-22T15:30:00",
  "updated_at": "2023-04-22T15:30:00"
}
```

### 6.5. HyperparamSearchResponse
```json
{
  "id": 654,
  "name": "Busca YOLOv8",
  "description": "Otimização de hiperparâmetros",
  "dataset_id": 123,
  "search_space": {
    "batch_size": {"min": 8, "max": 32},
    "lr0": {"min": 0.001, "max": 0.1}
  },
  "iterations": 5,
  "status": "running",
  "best_params": {
    "batch_size": 16,
    "lr0": 0.05
  },
  "best_metrics": {
    "map50": 0.756,
    "map": 0.512
  },
  "trials_data": [
    {
      "trial": 1,
      "params": {"batch_size": 16, "lr0": 0.01},
      "metrics": {"map50": 0.732, "map": 0.489}
    }
  ],
  "current_iteration": 1,
  "iterations_completed": 1,
  "created_at": "2023-04-21T14:00:00",
  "updated_at": "2023-04-21T14:05:00",
  "started_at": "2023-04-21T14:00:30",
  "completed_at": null
}
```

## 7. Fluxos de Trabalho

### 7.1. Criar e Treinar um Modelo

1. **Criar Dataset**:
   - Enviar requisição POST para `/datasets/`
   - Salvar o `id` do dataset

2. **Fazer Upload de Imagens**:
   - Para cada imagem, enviar POST para `/images/` com `dataset_id`

3. **Iniciar Treinamento**:
   - Enviar requisição POST para `/training/` com `dataset_id` e parâmetros
   - Salvar o `id` da sessão de treinamento

4. **Monitorar Progresso**:
   - Conectar ao WebSocket `ws://seu-servidor:porta/api/v1/training/ws/{session_id}`
   - Processar atualizações de progresso
   - Atualizar interface com gráficos e estatísticas

5. **Finalização**:
   - Quando receber status `completed` no WebSocket, o modelo está pronto
   - Obter detalhes finais via GET `/training/{session_id}`

### 7.2. Otimização de Hiperparâmetros

1. **Configurar Busca**:
   - Enviar requisição POST para `/hyperparams/` com `dataset_id` e espaço de busca
   - Salvar o `id` da busca

2. **Monitorar Progresso**:
   - Conectar ao WebSocket `ws://seu-servidor:porta/api/v1/hyperparams/ws/{search_id}`
   - Processar atualizações sobre trials e épocas
   - Atualizar interface com progresso total, gráficos e tabelas

3. **Finalização**:
   - Quando receber status `completed` no WebSocket, a busca terminou
   - Utilizar `best_params` para iniciar um treinamento otimizado

### 7.3. Fazer Inferência

1. **Selecionar Modelo**:
   - Obter lista via GET `/models/`
   - Selecionar modelo para inferência

2. **Enviar Imagem**:
   - Enviar POST para `/inference/` com arquivo de imagem e `model_id`

3. **Processar Resultados**:
   - Exibir a imagem com as detecções usando as coordenadas recebidas
   - Mostrar classes e confiança das detecções

## 8. Formato das Mensagens WebSocket

### 8.1. Monitoramento de Treinamento

O serviço envia atualizações a cada 100ms (0.1 segundos) e verifica:
- Progresso da época atual
- Métricas de treinamento
- Estado da tarefa (em execução, concluído, erro)

As mensagens são estruturadas conforme:

- **Progresso**: Inclui `current_epoch`, `total_epochs`, `percent_complete` e `progress_type`
- **Métricas**: Inclui valores como `loss`, `map50`, `map`, `precision`, `recall`
- **Status**: Um dos valores: `training`, `completed`, `failed`, `error`

### 8.2. Monitoramento de Busca de Hiperparâmetros

Similar ao treinamento, mas acrescenta:
- **Trials**: Lista de tentativas concluídas com parâmetros e resultados
- **Current Trial**: Informações sobre o trial atual
- **Best Params/Metrics**: Melhores parâmetros e métricas encontrados até o momento

## 9. Implementação no Flutter

### 9.1. Configuração Básica

```dart
class ApiConfig {
  static const String baseUrl = 'http://seu-servidor:porta/api/v1';
  static const String wsBaseUrl = 'ws://seu-servidor:porta/api/v1';
  
  // Headers padrão
  static Map<String, String> get headers => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };
}
```

### 9.2. Cliente WebSocket para Treinamento

```dart
class TrainingWebSocketClient {
  WebSocketChannel? _channel;
  StreamController<TrainingUpdate> _controller = StreamController.broadcast();
  Stream<TrainingUpdate> get updates => _controller.stream;
  
  void connect(int sessionId) {
    // Fechar conexão existente se houver
    disconnect();
    
    // Conectar ao WebSocket
    _channel = WebSocketChannel.connect(
      Uri.parse('${ApiConfig.wsBaseUrl}/training/ws/$sessionId'),
    );
    
    // Processar mensagens
    _channel!.stream.listen(
      (message) {
        Map<String, dynamic> data = jsonDecode(message);
        
        // Ignorar heartbeats
        if (data['type'] == 'heartbeat') return;
        
        // Processar estado inicial
        if (data['type'] == 'initial_state') {
          _controller.add(TrainingUpdate.fromInitialState(data['data']));
          return;
        }
        
        // Processar atualizações normais
        _controller.add(TrainingUpdate.fromJson(data));
      },
      onError: (error) {
        print('Erro WebSocket: $error');
        _controller.addError(error);
        // Tentar reconectar após erro
        Future.delayed(Duration(seconds: 3), () => connect(sessionId));
      },
      onDone: () {
        print('Conexão WebSocket fechada');
      },
    );
    
    // Enviar acknowledge ao estabelecer conexão
    _channel!.sink.add(jsonEncode({
      'type': 'acknowledge',
      'message': 'Cliente conectado'
    }));
  }
  
  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
  
  void dispose() {
    disconnect();
    _controller.close();
  }
}
```

## 10. Considerações Finais

### 10.1. Taxa de Atualização
O sistema foi otimizado para fornecer atualizações frequentes (10 vezes por segundo nos WebSockets) para garantir uma experiência de usuário responsiva.

### 10.2. Otimizações de Rede
As mensagens WebSocket foram projetadas para serem leves e conterem apenas as informações necessárias para atualizar a interface do usuário.

### 10.3. Manuseio de Erros
Implemente tratamento de erros adequado no cliente para lidar com desconexões e outros problemas de rede.

### 10.4. Heartbeats
O sistema envia heartbeats a cada 30 segundos para manter a conexão ativa e permitir a detecção rápida de problemas de conectividade. 
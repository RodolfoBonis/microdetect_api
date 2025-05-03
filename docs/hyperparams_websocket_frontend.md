# Integração de Busca de Hiperparâmetros e WebSocket para Frontend Flutter

Este documento descreve como integrar o frontend Flutter com a API de busca de hiperparâmetros e o WebSocket de monitoramento, usando como referência o fluxo automatizado do arquivo `test_full_flow.py`.

---

## 1. Criando uma Busca de Hiperparâmetros

### Endpoint HTTP

- **POST** `/api/v1/hyperparams/`

#### Payload de exemplo:
```json
{
  "dataset_id": 1,
  "name": "Busca Teste 20240503_153000",
  "description": "Busca de hiperparâmetros para testes automatizados",
  "search_space": {
    "model_type": "yolov8",
    "model_size": ["n", "s", "m"],
    "imgsz": [640, 1280],
    "optimizer": ["Adam", "SGD"],
    "device": "cpu",
    "epochs": {"min": 10, "max": 50},
    "batch_size": {"min": 8, "max": 32},
    "learning_rate": {"min": 0.0001, "max": 0.01}
  }
}
```

#### Resposta de sucesso:
```json
{
  "success": true,
  "data": {
    "id": 1,
    "status": "pending",
    "name": "Busca Teste 20240503_153000",
    "dataset_id": 1,
    "search_space": {
      "model_type": "yolov8",
      "model_size": ["n", "s", "m"],
      "imgsz": [640, 1280],
      "optimizer": ["Adam", "SGD"],
      "device": "cpu",
      "epochs": {"min": 10, "max": 50},
      "batch_size": {"min": 8, "max": 32},
      "learning_rate": {"min": 0.0001, "max": 0.01}
    },
    "iterations": 0,
    "description": "Busca de hiperparâmetros para testes automatizados",
    "best_params": {},
    "best_metrics": {},
    "trials_data": [],
    "created_at": "2025-05-03T15:00:00",
    "updated_at": "2025-05-03T15:00:00"
  }
}
```

- O campo `id` retornado será usado para monitorar a busca via WebSocket.

---

## 2. Monitorando o Progresso via WebSocket

### Endpoint WebSocket

- **URL:** `ws://<host>:<port>/api/v1/hyperparams/ws/{search_id}`
  - Exemplo: `ws://localhost:8000/api/v1/hyperparams/ws/1`

### Fluxo de Mensagens

1. **Conexão:**
   - Ao conectar, o servidor envia o estado inicial da busca:
   ```json
   {
     "type": "initial_state",
     "data": {
       "id": 1,
       "status": "running",
       "created_at": "2025-05-03T15:00:00",
       "updated_at": "2025-05-03T15:00:00",
       "dataset_id": 1,
       "search_space": {
         "model_type": "yolov8",
         "model_size": ["n", "s", "m"],
         "imgsz": [640, 1280],
         "optimizer": ["Adam", "SGD"],
         "device": "cpu",
         "epochs": {"min": 10, "max": 50},
         "batch_size": {"min": 8, "max": 32},
         "learning_rate": {"min": 0.0001, "max": 0.01}
       },
       "iterations": 432,
       "best_params": {},
       "best_metrics": {},
       "trials_data": []
     }
   }
   ```
2. **Confirmação:**
   - O cliente deve responder com:
   ```json
   { "type": "acknowledge" }
   ```
3. **Atualizações de Progresso:**
   - O servidor envia mensagens de progresso durante a execução:
   ```json
   {
     "status": "running",
     "trials": [
       {
         "params": {
           "model_type": "yolov8",
           "model_size": "n",
           "imgsz": 640,
           "optimizer": "Adam",
           "device": "cpu",
           "epochs": 10,
           "batch": 8,
           "lr0": 0.001
         },
         "metrics": {
           "epochs": 10,
           "best_epoch": 8,
           "best_map50": 0.82,
           "best_map": 0.78,
           "final_map50": 0.80,
           "final_map": 0.77,
           "train_time": 120.5,
           "val_time": 10.2,
           "precision": 0.85,
           "recall": 0.80,
           "f1_score": 0.82,
           "best_precision": 0.87,
           "best_recall": 0.82,
           "best_f1_score": 0.84
         }
       }
     ],
     "best_params": {
       "model_type": "yolov8",
       "model_size": "n",
       "imgsz": 640,
       "optimizer": "Adam",
       "device": "cpu",
       "epochs": 10,
       "batch": 8,
       "lr0": 0.001
     },
     "best_metrics": {
       "epochs": 10,
       "best_epoch": 8,
       "best_map50": 0.82,
       "best_map": 0.78,
       "final_map50": 0.80,
       "final_map": 0.77,
       "train_time": 120.5,
       "val_time": 10.2,
       "precision": 0.85,
       "recall": 0.80,
       "f1_score": 0.82,
       "best_precision": 0.87,
       "best_recall": 0.82,
       "best_f1_score": 0.84
     },
     "current_trial": 2,
     "total_trials": 432,
     "progress": {
       "trial": 2,
       "total_trials": 432,
       "current_epoch": 9,
       "total_epochs": 10,
       "percent_complete": 20
     },
     "current_trial_info": {
       "epoch": 9,
       "total_epochs": 10,
       "progress_type": "epoch_in_trial",
       "map50": 0.80,
       "map50_95": 0.75,
       "precision": 0.85,
       "recall": 0.80,
       "f1_score": 0.82,
       "current_trial": 2,
       "total_trials": 432
     }
   }
   ```
   - O campo `progress` mostra o andamento geral da busca.
   - O campo `current_trial_info` mostra o progresso detalhado do trial atual.
4. **Heartbeat:**
   - O servidor pode enviar mensagens de heartbeat para manter a conexão viva:
   ```json
   { "type": "heartbeat", "time": "2025-05-03T15:01:00" }
   ```
5. **Conclusão:**
   - Quando a busca termina:
   ```json
   {
     "status": "completed",
     "best_params": {
       "model_type": "yolov8",
       "model_size": "n",
       "imgsz": 640,
       "optimizer": "Adam",
       "device": "cpu",
       "epochs": 10,
       "batch": 8,
       "lr0": 0.001
     },
     "best_metrics": {
       "epochs": 10,
       "best_epoch": 8,
       "best_map50": 0.82,
       "best_map": 0.78,
       "final_map50": 0.80,
       "final_map": 0.77,
       "train_time": 120.5,
       "val_time": 10.2,
       "precision": 0.85,
       "recall": 0.80,
       "f1_score": 0.82,
       "best_precision": 0.87,
       "best_recall": 0.82,
       "best_f1_score": 0.84
     },
     "message": "Busca concluída com sucesso"
   }
   ```

---

## 3. Obtendo o Resultado Final

- **GET** `/api/v1/hyperparams/{search_id}`
- Retorna os melhores hiperparâmetros encontrados e o histórico dos trials.

#### Exemplo de resposta:
```json
{
  "success": true,
  "data": {
    "id": 1,
    "status": "completed",
    "best_params": {
      "model_type": "yolov8",
      "model_size": "n",
      "imgsz": 640,
      "optimizer": "Adam",
      "device": "cpu",
      "epochs": 10,
      "batch": 8,
      "lr0": 0.001
    },
    "best_metrics": {
      "epochs": 10,
      "best_epoch": 8,
      "best_map50": 0.82,
      "best_map": 0.78,
      "final_map50": 0.80,
      "final_map": 0.77,
      "train_time": 120.5,
      "val_time": 10.2,
      "precision": 0.85,
      "recall": 0.80,
      "f1_score": 0.82,
      "best_precision": 0.87,
      "best_recall": 0.82,
      "best_f1_score": 0.84
    },
    "trials_data": [
      {
        "params": {
          "model_type": "yolov8",
          "model_size": "n",
          "imgsz": 640,
          "optimizer": "Adam",
          "device": "cpu",
          "epochs": 10,
          "batch": 8,
          "lr0": 0.001
        },
        "metrics": {
          "epochs": 10,
          "best_epoch": 8,
          "best_map50": 0.82,
          "best_map": 0.78,
          "final_map50": 0.80,
          "final_map": 0.77,
          "train_time": 120.5,
          "val_time": 10.2,
          "precision": 0.85,
          "recall": 0.80,
          "f1_score": 0.82,
          "best_precision": 0.87,
          "best_recall": 0.82,
          "best_f1_score": 0.84
        }
      },
      {
        "params": {
          "model_type": "yolov8",
          "model_size": "s",
          "imgsz": 640,
          "optimizer": "SGD",
          "device": "cpu",
          "epochs": 10,
          "batch": 8,
          "lr0": 0.001
        },
        "metrics": {
          "epochs": 10,
          "best_epoch": 7,
          "best_map50": 0.80,
          "best_map": 0.76,
          "final_map50": 0.78,
          "final_map": 0.75,
          "train_time": 118.0,
          "val_time": 10.0,
          "precision": 0.83,
          "recall": 0.78,
          "f1_score": 0.80,
          "best_precision": 0.85,
          "best_recall": 0.80,
          "best_f1_score": 0.82
        }
      }
    ]
  }
}
```

---

## 4. Dicas para o Frontend Flutter

- **Conexão WebSocket:** Use um pacote como `web_socket_channel` para conectar e ouvir as mensagens.
- **Confirmação:** Sempre envie `{ "type": "acknowledge" }` após receber o estado inicial.
- **Progresso:** Atualize a UI conforme os campos `progress` e `current_trial_info` mudam.
- **Heartbeat:** Ignore ou use para mostrar que a conexão está ativa.
- **Conclusão:** Quando `status` for `completed`, busque os melhores parâmetros via HTTP ou use os dados da última mensagem.
- **Timeouts:** Considere mostrar mensagens de "em andamento" e tratar desconexões automáticas.

---

## 5. Exemplo de Fluxo no Flutter (pseudo-código)

```dart
final ws = WebSocketChannel.connect(Uri.parse('ws://localhost:8000/api/v1/hyperparams/ws/$searchId'));
ws.stream.listen((message) {
  final data = jsonDecode(message);
  if (data['type'] == 'initial_state') {
    ws.sink.add(jsonEncode({ 'type': 'acknowledge' }));
  } else if (data['type'] == 'heartbeat') {
    // opcional: atualizar indicador de conexão
  } else if (data['status'] == 'running') {
    // atualizar UI com progresso
  } else if (data['status'] == 'completed') {
    // mostrar melhores hiperparâmetros
  }
});
```

---

## 6. Observações

- O número de trials pode ser grande, então atualize a UI de forma eficiente.
- As métricas detalhadas podem aparecer apenas após a validação de cada trial.
- Sempre trate desconexões e tente reconectar se necessário.

---

**Dúvidas ou sugestões? Consulte o arquivo `test_full_flow.py` para exemplos completos de uso.** 
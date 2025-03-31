# Guia de Integração Frontend - Módulo de Hiperparâmetros e Treinamento

Este documento descreve como integrar o frontend Flutter com os módulos de busca de hiperparâmetros e treinamento da API MicroDetect.

## 1. Visão Geral da Arquitetura

A comunicação entre o frontend e o backend para treinamento e busca de hiperparâmetros tem dois componentes principais:

1. **Endpoints REST**: Para operações assíncronas como iniciar treinamento, listar sessões, etc.
2. **WebSockets**: Para comunicação em tempo real sobre progresso de treinamento, uso de recursos e métricas.

## 2. Busca de Hiperparâmetros

### 2.1. Iniciar Busca de Hiperparâmetros

```dart
Future<HyperparamSearch> startHyperparamSearch(HyperparamSearchData data) async {
  final response = await http.post(
    Uri.parse('${apiBaseUrl}/api/v1/hyperparams/'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(data.toJson()),
  );
  
  if (response.statusCode == 200) {
    return HyperparamSearch.fromJson(jsonDecode(response.body)['data']);
  } else {
    throw Exception('Falha ao iniciar busca de hiperparâmetros: ${response.body}');
  }
}
```

Exemplo de dados para a busca:

```dart
final searchData = {
  'name': 'Busca otimizada YOLOv8',
  'dataset_id': 123,
  'search_space': {
    'model_type': 'yolov8',
    'model_size': ['n', 's', 'm'],
    'batch_size': {'min': 8, 'max': 32},
    'imgsz': [416, 512, 640],
    'epochs': 10,  // Reduzir épocas para busca mais rápida
    'optimizer': ['Adam', 'SGD'],
    'lr0': {'min': 0.001, 'max': 0.01},
  },
  'iterations': 5,  // Número de modelos a serem testados
  'description': 'Busca automática para encontrar melhores hiperparâmetros'
};
```

### 2.2. Monitorar Progresso da Busca via WebSocket

```dart
void connectToHyperparamSearchWebSocket(int searchId) {
  final ws = WebSocketChannel.connect(
    Uri.parse('${wsBaseUrl}/api/v1/hyperparams/ws/${searchId}'),
  );
  
  ws.stream.listen(
    (message) {
      final data = jsonDecode(message);
      
      // Atualizar UI com progresso de busca
      setState(() {
        searchStatus = data['status'];
        currentIteration = data['trials_data'].length;
        totalIterations = data['iterations'];
        
        if (data['best_params'] != null) {
          bestParams = data['best_params'];
          bestMetrics = data['best_metrics'];
        }
        
        // Se completado, mostrar botão para ver modelo final
        if (data['status'] == 'completed' && data['training_session_id'] != null) {
          finalModelId = data['training_session_id'];
        }
      });
    },
    onDone: () {
      // Conexão fechada
      print('Conexão WebSocket fechada');
    },
    onError: (error) {
      print('Erro na conexão WebSocket: $error');
    }
  );
  
  // Armazenar para fechar depois
  _wsChannel = ws;
}

@override
void dispose() {
  _wsChannel?.sink.close();
  super.dispose();
}
```

### 2.3. Obter Lista de Buscas

```dart
Future<List<HyperparamSearch>> listHyperparamSearches({
  int? datasetId,
  String? status,
}) async {
  final queryParams = <String, String>{};
  if (datasetId != null) queryParams['dataset_id'] = datasetId.toString();
  if (status != null) queryParams['status'] = status;
  
  final response = await http.get(
    Uri.parse('${apiBaseUrl}/api/v1/hyperparams/').replace(queryParameters: queryParams),
  );
  
  if (response.statusCode == 200) {
    final data = jsonDecode(response.body)['data'] as List;
    return data.map((json) => HyperparamSearch.fromJson(json)).toList();
  } else {
    throw Exception('Falha ao obter lista de buscas: ${response.body}');
  }
}
```

### 2.4. Obter Detalhes de uma Busca

```dart
Future<HyperparamSearch> getHyperparamSearch(int searchId) async {
  final response = await http.get(
    Uri.parse('${apiBaseUrl}/api/v1/hyperparams/${searchId}'),
  );
  
  if (response.statusCode == 200) {
    return HyperparamSearch.fromJson(jsonDecode(response.body)['data']);
  } else {
    throw Exception('Falha ao obter detalhes da busca: ${response.body}');
  }
}
```

## 3. Treinamento de Modelos

### 3.1. Iniciar Treinamento

```dart
Future<TrainingSession> startTraining(TrainingData data) async {
  final response = await http.post(
    Uri.parse('${apiBaseUrl}/api/v1/training/'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(data.toJson()),
  );
  
  if (response.statusCode == 200) {
    return TrainingSession.fromJson(jsonDecode(response.body)['data']);
  } else {
    throw Exception('Falha ao iniciar treinamento: ${response.body}');
  }
}
```

Exemplo de dados para o treinamento:

```dart
final trainingData = {
  'name': 'YOLOv8n Microorganismos',
  'model_type': 'yolov8',
  'model_version': 'n',  // n=nano, s=small, m=medium, l=large, x=xlarge
  'dataset_id': 123,
  'description': 'Modelo para detectar microorganismos em imagens microscópicas',
  'hyperparameters': {
    'batch_size': 16,
    'imgsz': 640,
    'epochs': 100,
    'device': 'auto',  // auto, 0 (primeira GPU), cpu
    'optimizer': 'auto',
    'lr0': 0.01,
    'patience': 50,  // early stopping
  }
};
```

### 3.2. Monitorar Progresso de Treinamento via WebSocket

```dart
void connectToTrainingWebSocket(int sessionId) {
  final ws = WebSocketChannel.connect(
    Uri.parse('${wsBaseUrl}/api/v1/training/ws/${sessionId}'),
  );
  
  ws.stream.listen(
    (message) {
      final data = jsonDecode(message);
      
      // Verificar se é um relatório final
      if (data['type'] == 'final_report') {
        setState(() {
          finalReport = data['data'];
          // Mostrar tela de relatório
          showReportView = true;
        });
        return;
      }
      
      // Caso contrário, é progresso de treinamento
      if (data['metrics'] != null) {
        setState(() {
          currentEpoch = data['current_epoch'];
          totalEpochs = data['total_epochs'];
          loss = data['metrics']['loss'];
          
          // Dados de validação (podem ser null em algumas épocas)
          if (data['metrics']['map50'] != null) {
            map50 = data['metrics']['map50'];
          }
          
          // Dados de recursos
          if (data['metrics']['resources'] != null) {
            cpuUsage = data['metrics']['resources']['cpu_percent'];
            memoryUsage = data['metrics']['resources']['memory_percent'];
            gpuUsage = data['metrics']['resources']['gpu_percent'];
          }
          
          // Atualizar gráficos de progresso
          addDataPointToChart(currentEpoch, loss, map50);
        });
      } else if (data['status'] != null) {
        // Atualização de status
        setState(() {
          trainingStatus = data['status'];
        });
      }
    },
    onDone: () {
      // Conexão fechada
      print('Conexão WebSocket fechada');
    },
    onError: (error) {
      print('Erro na conexão WebSocket: $error');
    }
  );
  
  // Armazenar para fechar depois
  _wsChannel = ws;
}
```

### 3.3. Obter Relatório de Treinamento

```dart
Future<TrainingReport> getTrainingReport(int sessionId) async {
  final response = await http.get(
    Uri.parse('${apiBaseUrl}/api/v1/training/${sessionId}/report'),
  );
  
  if (response.statusCode == 200) {
    return TrainingReport.fromJson(jsonDecode(response.body)['data']);
  } else {
    throw Exception('Falha ao obter relatório de treinamento: ${response.body}');
  }
}
```

## 4. Exibindo Relatórios e Gráficos

### 4.1. Gráficos de Métricas de Treinamento

Para exibir gráficos interativos com o progresso do treinamento, você pode usar a biblioteca `fl_chart`:

```dart
Widget buildTrainingChart() {
  return LineChart(
    LineChartData(
      gridData: FlGridData(show: true),
      titlesData: FlTitlesData(show: true),
      borderData: FlBorderData(show: true),
      lineBarsData: [
        // Linha de loss
        LineChartBarData(
          spots: lossHistory.asMap().entries
              .map((entry) => FlSpot(entry.key.toDouble(), entry.value))
              .toList(),
          isCurved: true,
          colors: [Colors.red],
          barWidth: 2,
          dotData: FlDotData(show: false),
        ),
        // Linha de mAP50
        LineChartBarData(
          spots: map50History.asMap().entries
              .map((entry) => FlSpot(entry.key.toDouble(), entry.value))
              .toList(),
          isCurved: true,
          colors: [Colors.blue],
          barWidth: 2,
          dotData: FlDotData(show: false),
        ),
      ],
    ),
  );
}
```

### 4.2. Matriz de Confusão

Para visualizar a matriz de confusão do relatório final:

```dart
Widget buildConfusionMatrix(List<List<int>> matrix, List<String> classNames) {
  return Container(
    margin: EdgeInsets.all(16),
    child: Column(
      children: [
        Text('Matriz de Confusão', style: TextStyle(fontWeight: FontWeight.bold)),
        SizedBox(height: 16),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: DataTable(
            columns: [
              DataColumn(label: Text('')), // Célula vazia no canto superior esquerdo
              ...classNames.map((name) => DataColumn(label: Text(name))),
            ],
            rows: List.generate(
              matrix.length,
              (i) => DataRow(
                cells: [
                  DataCell(Text(classNames[i])), // Nome da classe
                  ...matrix[i].map((value) => DataCell(
                    Container(
                      padding: EdgeInsets.all(8),
                      color: Colors.blue.withOpacity(value / (matrix[i].reduce((a, b) => a + b) + 1)),
                      child: Text('$value', style: TextStyle(color: Colors.white)),
                    )
                  )),
                ],
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
```

### 4.3. Desempenho por Classe

Para exibir o desempenho de cada classe:

```dart
Widget buildClassPerformance(List<ClassPerformance> performances) {
  return Container(
    margin: EdgeInsets.all(16),
    child: Column(
      children: [
        Text('Desempenho por Classe', style: TextStyle(fontWeight: FontWeight.bold)),
        SizedBox(height: 16),
        ListView.builder(
          shrinkWrap: true,
          physics: NeverScrollableScrollPhysics(),
          itemCount: performances.length,
          itemBuilder: (context, index) {
            final perf = performances[index];
            return Card(
              margin: EdgeInsets.symmetric(vertical: 8),
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('${perf.className} (ID: ${perf.classId})', 
                      style: TextStyle(fontWeight: FontWeight.bold)),
                    SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildMetricItem('Precisão', perf.precision),
                        _buildMetricItem('Recall', perf.recall),
                        _buildMetricItem('F1-Score', perf.f1Score),
                      ],
                    ),
                    SizedBox(height: 8),
                    Text('Exemplos: ${perf.examplesCount}'),
                  ],
                ),
              ),
            );
          },
        ),
      ],
    ),
  );
}

Widget _buildMetricItem(String label, double value) {
  return Column(
    children: [
      Text(label),
      SizedBox(height: 4),
      Text('${(value * 100).toStringAsFixed(1)}%', 
        style: TextStyle(fontWeight: FontWeight.bold)),
    ],
  );
}
```

## 5. Modelos de Dados Sugeridos

### 5.1. HyperparamSearch

```dart
class HyperparamSearch {
  final int id;
  final String name;
  final String description;
  final String status;
  final int datasetId;
  final Map<String, dynamic> searchSpace;
  final Map<String, dynamic>? bestParams;
  final Map<String, dynamic>? bestMetrics;
  final List<Map<String, dynamic>>? trialsData;
  final int iterations;
  final DateTime createdAt;
  final DateTime? startedAt;
  final DateTime? completedAt;
  final int? trainingSessionId;

  HyperparamSearch({
    required this.id,
    required this.name,
    required this.description,
    required this.status,
    required this.datasetId,
    required this.searchSpace,
    this.bestParams,
    this.bestMetrics,
    this.trialsData,
    required this.iterations,
    required this.createdAt,
    this.startedAt,
    this.completedAt,
    this.trainingSessionId,
  });

  factory HyperparamSearch.fromJson(Map<String, dynamic> json) {
    return HyperparamSearch(
      id: json['id'],
      name: json['name'],
      description: json['description'] ?? '',
      status: json['status'],
      datasetId: json['dataset_id'],
      searchSpace: json['search_space'] ?? {},
      bestParams: json['best_params'],
      bestMetrics: json['best_metrics'],
      trialsData: json['trials_data'] != null 
          ? List<Map<String, dynamic>>.from(json['trials_data']) 
          : null,
      iterations: json['iterations'] ?? 5,
      createdAt: DateTime.parse(json['created_at']),
      startedAt: json['started_at'] != null ? DateTime.parse(json['started_at']) : null,
      completedAt: json['completed_at'] != null ? DateTime.parse(json['completed_at']) : null,
      trainingSessionId: json['training_session_id'],
    );
  }
}
```

### 5.2. TrainingSession

```dart
class TrainingSession {
  final int id;
  final String name;
  final String description;
  final String status;
  final String modelType;
  final String modelVersion;
  final int datasetId;
  final Map<String, dynamic> hyperparameters;
  final Map<String, dynamic>? metrics;
  final DateTime createdAt;
  final DateTime? startedAt;
  final DateTime? completedAt;

  TrainingSession({
    required this.id,
    required this.name,
    required this.description,
    required this.status,
    required this.modelType,
    required this.modelVersion,
    required this.datasetId,
    required this.hyperparameters,
    this.metrics,
    required this.createdAt,
    this.startedAt,
    this.completedAt,
  });

  factory TrainingSession.fromJson(Map<String, dynamic> json) {
    return TrainingSession(
      id: json['id'],
      name: json['name'],
      description: json['description'] ?? '',
      status: json['status'],
      modelType: json['model_type'],
      modelVersion: json['model_version'],
      datasetId: json['dataset_id'],
      hyperparameters: json['hyperparameters'] ?? {},
      metrics: json['metrics'],
      createdAt: DateTime.parse(json['created_at']),
      startedAt: json['started_at'] != null ? DateTime.parse(json['started_at']) : null,
      completedAt: json['completed_at'] != null ? DateTime.parse(json['completed_at']) : null,
    );
  }
}
```

### 5.3. TrainingReport

```dart
class TrainingReport {
  final int id;
  final int trainingSessionId;
  final String modelName;
  final int datasetId;
  final List<TrainingMetrics> metricsHistory;
  final List<List<int>> confusionMatrix;
  final List<ClassPerformance> classPerformance;
  final Map<String, dynamic> finalMetrics;
  final ResourceUsage resourceUsageAvg;
  final ResourceUsage resourceUsageMax;
  final Map<String, dynamic> hyperparameters;
  final int trainImagesCount;
  final int valImagesCount;
  final int testImagesCount;
  final int trainingTimeSeconds;
  final double modelSizeMb;
  final DateTime createdAt;

  TrainingReport({
    required this.id,
    required this.trainingSessionId,
    required this.modelName,
    required this.datasetId,
    required this.metricsHistory,
    required this.confusionMatrix,
    required this.classPerformance,
    required this.finalMetrics,
    required this.resourceUsageAvg,
    required this.resourceUsageMax,
    required this.hyperparameters,
    required this.trainImagesCount,
    required this.valImagesCount,
    required this.testImagesCount,
    required this.trainingTimeSeconds,
    required this.modelSizeMb,
    required this.createdAt,
  });

  factory TrainingReport.fromJson(Map<String, dynamic> json) {
    return TrainingReport(
      id: json['id'],
      trainingSessionId: json['training_session_id'],
      modelName: json['model_name'],
      datasetId: json['dataset_id'],
      metricsHistory: json['metrics_history'] != null 
          ? List<Map<String, dynamic>>.from(json['metrics_history'])
              .map((m) => TrainingMetrics.fromJson(m)).toList() 
          : [],
      confusionMatrix: json['confusion_matrix'] != null 
          ? List<List<int>>.from(json['confusion_matrix']
              .map((row) => List<int>.from(row))) 
          : [],
      classPerformance: json['class_performance'] != null 
          ? List<Map<String, dynamic>>.from(json['class_performance'])
              .map((c) => ClassPerformance.fromJson(c)).toList() 
          : [],
      finalMetrics: json['final_metrics'] ?? {},
      resourceUsageAvg: ResourceUsage.fromJson(json['resource_usage_avg'] ?? {}),
      resourceUsageMax: ResourceUsage.fromJson(json['resource_usage_max'] ?? {}),
      hyperparameters: json['hyperparameters'] ?? {},
      trainImagesCount: json['train_images_count'] ?? 0,
      valImagesCount: json['val_images_count'] ?? 0,
      testImagesCount: json['test_images_count'] ?? 0,
      trainingTimeSeconds: json['training_time_seconds'] ?? 0,
      modelSizeMb: json['model_size_mb'] ?? 0.0,
      createdAt: DateTime.parse(json['created_at']),
    );
  }
}

class ClassPerformance {
  final int classId;
  final String className;
  final double precision;
  final double recall;
  final double f1Score;
  final int support;
  final int examplesCount;

  ClassPerformance({
    required this.classId,
    required this.className,
    required this.precision,
    required this.recall,
    required this.f1Score,
    required this.support,
    required this.examplesCount,
  });

  factory ClassPerformance.fromJson(Map<String, dynamic> json) {
    return ClassPerformance(
      classId: json['class_id'],
      className: json['class_name'],
      precision: json['precision'] ?? 0.0,
      recall: json['recall'] ?? 0.0,
      f1Score: json['f1_score'] ?? 0.0,
      support: json['support'] ?? 0,
      examplesCount: json['examples_count'] ?? 0,
    );
  }
}

class ResourceUsage {
  final double cpuPercent;
  final double memoryPercent;
  final double? gpuPercent;
  final double? gpuMemoryPercent;

  ResourceUsage({
    required this.cpuPercent,
    required this.memoryPercent,
    this.gpuPercent,
    this.gpuMemoryPercent,
  });

  factory ResourceUsage.fromJson(Map<String, dynamic> json) {
    return ResourceUsage(
      cpuPercent: json['cpu_percent'] ?? 0.0,
      memoryPercent: json['memory_percent'] ?? 0.0,
      gpuPercent: json['gpu_percent'],
      gpuMemoryPercent: json['gpu_memory_percent'],
    );
  }
}

class TrainingMetrics {
  final int epoch;
  final double loss;
  final double? valLoss;
  final double? map50;
  final double? map;
  final double? precision;
  final double? recall;
  final ResourceUsage? resources;

  TrainingMetrics({
    required this.epoch,
    required this.loss,
    this.valLoss,
    this.map50,
    this.map,
    this.precision,
    this.recall,
    this.resources,
  });

  factory TrainingMetrics.fromJson(Map<String, dynamic> json) {
    return TrainingMetrics(
      epoch: json['epoch'] ?? 0,
      loss: json['loss'] ?? 0.0,
      valLoss: json['val_loss'],
      map50: json['map50'],
      map: json['map'],
      precision: json['precision'],
      recall: json['recall'],
      resources: json['resources'] != null 
          ? ResourceUsage.fromJson(json['resources']) 
          : null,
    );
  }
}
```

## 6. Fluxo Completo

### 6.1. Busca de Hiperparâmetros

1. Usuário seleciona dataset e configura o espaço de busca de hiperparâmetros
2. Frontend envia requisição para iniciar busca
3. Backend inicia processo de busca em background
4. Frontend conecta ao WebSocket para monitorar progresso
5. Frontend exibe gráficos e métricas em tempo real
6. Quando busca termina, frontend mostra botão para visualizar modelo final ou iniciar treinamento com os melhores hiperparâmetros

### 6.2. Treinamento de Modelo

1. Usuário configura parâmetros de treinamento (manualmente ou a partir de busca)
2. Frontend envia requisição para iniciar treinamento
3. Backend inicia processo de treinamento em background
4. Frontend conecta ao WebSocket para monitorar progresso
5. Frontend exibe gráficos e métricas em tempo real de:
   - Loss e métricas de validação (mAP50, precisão, recall)
   - Uso de recursos (CPU, RAM, GPU)
   - Progresso de épocas (atual/total)
6. Quando treinamento termina, frontend recebe relatório final
7. Frontend exibe detalhes do relatório com visualizações interativas 
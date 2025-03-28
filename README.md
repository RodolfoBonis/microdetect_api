# MicroDetect Backend

Backend Python para a aplicação MicroDetect, responsável por gerenciar datasets, modelos e inferências.

## Estrutura do Projeto

```
python_backend/
├── app/                    # Código da aplicação
│   ├── api/               # Endpoints da API
│   ├── core/              # Configurações e utilitários
│   ├── database/          # Modelos e conexão com banco
│   ├── models/            # Modelos ML
│   ├── schemas/           # Esquemas Pydantic
│   ├── services/          # Serviços
│   └── utils/             # Utilitários
├── data/                  # Diretório para dados
│   ├── datasets/          # Datasets
│   ├── models/            # Modelos treinados
│   ├── gallery/           # Imagens capturadas
│   └── temp/              # Arquivos temporários
├── tests/                 # Testes
├── requirements.txt       # Dependências
└── start_backend.py       # Script de inicialização
```

## Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

## Instalação

1. Clone o repositório
2. Navegue até o diretório do backend:
   ```bash
   cd python_backend
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para iniciar o servidor:

```bash
python start_backend.py
```

O servidor estará disponível em `http://localhost:8000`

## API Endpoints

### Datasets
- `POST /api/v1/datasets/` - Criar novo dataset
- `GET /api/v1/datasets/` - Listar datasets
- `GET /api/v1/datasets/{id}` - Obter dataset específico

### Imagens
- `POST /api/v1/images/` - Upload de imagem
- `GET /api/v1/images/` - Listar imagens
- `GET /api/v1/images/{id}` - Obter imagem específica

### Anotações
- `POST /api/v1/annotations/` - Criar anotação
- `GET /api/v1/annotations/` - Listar anotações
- `GET /api/v1/annotations/{id}` - Obter anotação específica

### Treinamento
- `POST /api/v1/training/` - Iniciar sessão de treinamento
- `GET /api/v1/training/` - Listar sessões de treinamento
- `GET /api/v1/training/{id}` - Obter sessão específica

### Modelos
- `POST /api/v1/models/` - Criar novo modelo
- `GET /api/v1/models/` - Listar modelos
- `GET /api/v1/models/{id}` - Obter modelo específico

### Inferência
- `POST /api/v1/inference/` - Realizar inferência
- `GET /api/v1/inference/` - Listar resultados de inferência
- `GET /api/v1/inference/{id}` - Obter resultado específico

## Documentação da API

A documentação interativa da API está disponível em:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Integração com Flutter

O backend é iniciado automaticamente quando a aplicação Flutter é executada. O script `start_backend.py` é responsável por:

1. Instalar dependências necessárias
2. Iniciar o servidor FastAPI
3. Gerenciar o ciclo de vida do servidor

## Desenvolvimento

Para desenvolvimento local:

1. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o servidor em modo de desenvolvimento:
   ```bash
   python start_backend.py
   ```

## Testes

Para executar os testes:

```bash
pytest
```

## Contribuição

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request 

## Migrações de Banco de Dados

Este projeto utiliza o Alembic para gerenciar migrações de banco de dados. As migrações são executadas automaticamente durante a inicialização da API, garantindo que o esquema do banco de dados esteja sempre atualizado.

### Como Funcionam as Migrações

1. Quando a API inicia, ela executa automaticamente `alembic upgrade head` para aplicar todas as migrações pendentes.
2. Se ocorrer algum erro durante as migrações, a API tenta criar as tabelas diretamente usando SQLAlchemy como fallback.

### Criando Novas Migrações

Quando você fizer alterações nos modelos de dados (como adicionar/remover colunas, criar novas tabelas, etc.), você precisa criar uma nova migração. Use o script utilitário:

```bash
# No diretório raiz do projeto
python scripts/create_migration.py "Descrição da sua migração"
```

O script irá:
1. Gerar um arquivo de migração na pasta `alembic/versions/`
2. O arquivo conterá as alterações detectadas nos seus modelos

### Aplicando Migrações Manualmente

Normalmente as migrações são aplicadas automaticamente quando a API inicia, mas você pode aplicá-las manualmente:

```bash
# No diretório raiz do projeto
python scripts/apply_migrations.py
```

Ou diretamente com o Alembic:

```bash
# No diretório raiz do projeto
python -m alembic upgrade head
```

### Outras Operações com Alembic

- Verificar migrações pendentes:
  ```bash
  python -m alembic current
  ```

- Voltar para uma migração específica:
  ```bash
  python -m alembic downgrade <migration_id>
  ```

- Gerar uma migração (mesmo que não haja mudanças):
  ```bash
  python -m alembic revision -m "Descrição da migração"
  ```

## Como substituir o Pydantic

Este projeto foi adaptado para usar classes Python regulares em vez do Pydantic. Principais alterações:

1. Criada uma classe base `BaseSchema` em `microdetect/schemas/base.py` que fornece funcionalidades de:
   - Inicialização via `__init__`
   - Serialização para dicionário via método `dict()`
   - Conversão de objetos ORM para schemas via método `from_orm()`

2. Classes de esquema implementadas como classes regulares do Python que herdam de `BaseSchema`

3. Utilitários de serialização em `microdetect/utils/serializers.py` para:
   - Converter objetos Python para JSON
   - Construir respostas da API padronizadas

4. Os endpoints da API foram atualizados para usar os serializadores personalizados

### Vantagens desta abordagem:
- Sem dependências externas para validação/serialização
- Maior controle sobre o comportamento de serialização
- Classes mais simples e explícitas
- Maior flexibilidade na manipulação dos dados

### Exemplo de uso:
```python
# Criar um objeto a partir de dados
model = SimpleModelResponse(
    id=1,
    name="Modelo de teste",
    description="Um modelo para teste"
)

# Converter para dicionário
model_dict = model.dict()

# Criar a partir de um objeto ORM
from_db = SimpleModelResponse.from_orm(db_model)

# Construir resposta da API
from microdetect.utils.serializers import build_response
response = build_response(model)
``` 
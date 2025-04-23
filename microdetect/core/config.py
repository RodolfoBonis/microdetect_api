from pathlib import Path
from typing import Optional, Dict, Any, Set, List
import os
import dotenv
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings
import logging
import platform

logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente do arquivo .env se existir
dotenv.load_dotenv(".env")

class Settings(BaseSettings):
    # Diretórios
    BASE_DIR: str = os.path.expanduser("~/.microdetect")
    DATA_DIR: str = os.path.expanduser("~/.microdetect/data")
    DATASETS_DIR: str = os.path.expanduser("~/.microdetect/data/datasets")
    MODELS_DIR: str = os.path.expanduser("~/.microdetect/data/models")
    IMAGES_DIR: str = os.path.expanduser("~/.microdetect/data/images")
    GALLERY_DIR: str = os.path.expanduser("~/.microdetect/data/gallery")
    TEMP_DIR: str = os.path.expanduser("~/.microdetect/data/temp")
    STATIC_DIR: str = os.path.expanduser("~/.microdetect/data/static")
    TRAINING_DIR: str = os.path.expanduser("~/.microdetect/data/training")
    ANNOTATIONS_DIR: str = os.path.expanduser("~/.microdetect/data/annotations")
    EXPORTS_DIR: str = os.path.expanduser("~/.microdetect/data/exports")
    
    # Banco de dados
    DATABASE_URL: str = f"sqlite:///{os.path.expanduser('~/.microdetect/microdetect.db')}"
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MicroDetect API"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_DB: str = "0"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CUDA/GPU
    USE_CUDA: bool = False  # Será atualizado automaticamente na inicialização
    USE_MPS: bool = False   # Para Metal Performance Shaders no MacOS
    CUDA_DEVICE: str = "cuda:0"
    MPS_DEVICE: str = "mps"  # Dispositivo para MacOS
    FORCE_CPU: bool = False  # Se True, força o uso de CPU mesmo com GPU disponível
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    # Configurar o método de inicialização de processos para o Celery
    # 'spawn' é mais seguro no MacOS com MPS do que 'fork'
    MULTIPROCESSING_START_METHOD: str = "spawn" if platform.system() == "Darwin" else "fork" 
    
    # Treinamento
    DEFAULT_MODEL_TYPE: str = "yolov8"
    DEFAULT_MODEL_VERSION: str = "n"
    MAX_BATCH_SIZE: int = 32
    MAX_IMAGE_SIZE: int = 640
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env"
    }

# Criar instância das configurações
settings = Settings()

# Criar diretórios
for directory in [
    settings.BASE_DIR,
    settings.DATA_DIR,
    settings.DATASETS_DIR,
    settings.MODELS_DIR,
    settings.IMAGES_DIR,
    settings.GALLERY_DIR,
    settings.TEMP_DIR,
    settings.STATIC_DIR,
    settings.TRAINING_DIR,
    settings.ANNOTATIONS_DIR,
    settings.EXPORTS_DIR
]:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório criado: {path}")

# Atualizar USE_CUDA baseado na disponibilidade real de CUDA
try:
    import torch
    settings.USE_CUDA = torch.cuda.is_available() and not settings.FORCE_CPU
except ImportError:
    settings.USE_CUDA = False

# Configurações básicas
API_V1_STR: str = "/api/v1"
PROJECT_NAME: str = "MicroDetect API"

# Diretórios base
BASE_DIR: Path = Path(os.path.expanduser("~/.microdetect"))
APP_DIR: Path = BASE_DIR / "app"

# Diretório no home do usuário
HOME_DIR: Path = Path.home() / ".microdetect"

# Diretórios de dados (agora no ~/.microdetect)
DATA_DIR: Path = Path(os.path.expanduser("~/.microdetect/data"))
DATASETS_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/datasets"))
MODELS_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/models"))
IMAGES_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/images"))
GALLERY_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/gallery"))
TEMP_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/temp"))
STATIC_DIR: Path = Path(os.path.expanduser("~/.microdetect/data/static"))

# Diretórios específicos
ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
TRAINING_DIR: Path = DATA_DIR / "training"
EXPORTS_DIR: Path = DATA_DIR / "exports"

# Configurações do servidor
HOST: str = "0.0.0.0"
PORT: int = 8000
DEBUG: bool = False

# Configurações de segurança
SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Configurações de upload
MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES: Set[str] = field(default_factory=lambda: {"image/jpeg", "image/png", "image/tiff"})

# Configurações do banco de dados
DATABASE_URL: str = "sqlite:///microdetect.db"

# Redis settings
REDIS_HOST: str = "localhost"
REDIS_PORT: str = "6379"
REDIS_DB: str = "0"
REDIS_URL: str = "redis://localhost:6379/0"

# CUDA/GPU
USE_CUDA: bool = False  # Será atualizado automaticamente na inicialização
CUDA_DEVICE: str = "cuda:0"
FORCE_CPU: bool = False  # Se True, força o uso de CPU mesmo com GPU disponível

# Celery
CELERY_BROKER_URL: str = "amqp://microdetect:microdetect123@localhost:5672//"
CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

# Treinamento
DEFAULT_MODEL_TYPE: str = "yolov8"
DEFAULT_MODEL_VERSION: str = "n"
MAX_BATCH_SIZE: int = 32
MAX_IMAGE_SIZE: int = 640

# Configurações de processamento
MAX_WORKERS: int = os.cpu_count() or 4
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))

def create_directories(self):
    """Cria os diretórios necessários para a aplicação"""
    dirs = [
        self.BASE_DIR,
        self.DATA_DIR,
        self.IMAGES_DIR,
        self.ANNOTATIONS_DIR,
        self.TRAINING_DIR,
        self.MODELS_DIR,
        self.EXPORTS_DIR,
        self.TEMP_DIR
    ]
    
    for directory in dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório criado: {directory}")
        
def __init__(self):
    """Inicializa as configurações e cria diretórios necessários"""
    self.create_directories()

def model_post_init(self, *args, **kwargs):
    """Método chamado após a inicialização do modelo."""
    self.update_cuda_status()

def update_cuda_status(self):
    """Atualiza o status do CUDA/MPS."""
    try:
        import torch
        self.USE_CUDA = torch.cuda.is_available() and not self.FORCE_CPU
        
        # Verificar suporte a MPS (Metal Performance Shaders) no MacOS
        if platform.system() == "Darwin" and not self.FORCE_CPU:
            self.USE_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if self.USE_MPS:
                logger.info("Metal Performance Shaders (MPS) está disponível no MacOS")
        else:
            self.USE_MPS = False
                
    except ImportError:
        self.USE_CUDA = False
        self.USE_MPS = False
        
    def get_device(self):
        """Retorna o dispositivo apropriado (cuda, mps, cpu)"""
        if self.FORCE_CPU:
            return "cpu"
        elif self.USE_CUDA:
            return self.CUDA_DEVICE
        elif self.USE_MPS:
            return self.MPS_DEVICE
        else:
            return "cpu"
from pathlib import Path
from typing import Optional, Dict, Any, Set, List
import os
import dotenv
from dataclasses import dataclass, field

# Carregar variáveis de ambiente do arquivo .env se existir
dotenv.load_dotenv(".env")

@dataclass
class Settings:
    # Configurações básicas
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MicroDetect API"

    # Diretórios base
    BASE_DIR: Path = Path(os.path.expanduser("~/.microdetect"))
    APP_DIR: Path = BASE_DIR / "app"

    # Diretório no home do usuário
    HOME_DIR: Path = Path.home() / ".microdetect"

    # Diretórios de dados (agora no ~/.microdetect)
    DATA_DIR: Path = BASE_DIR / "data"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    MODELS_DIR: Path = DATA_DIR / "models"
    IMAGES_DIR: Path = DATA_DIR / "images"
    GALLERY_DIR: Path = DATA_DIR / "gallery"
    TEMP_DIR: Path = DATA_DIR / "temp"
    STATIC_DIR: Path = DATA_DIR / "static"

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
    DATABASE_URL: str = "sqlite:///{}".format(str(BASE_DIR / "microdetect.db"))
    
    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_DB = os.getenv("REDIS_DB", "0")
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

    def __post_init__(self):
        # Configurar DATABASE_URL se não foi definido por variável de ambiente
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"sqlite:///{self.HOME_DIR}/microdetect.db"
            
        # Criar diretórios necessários se não existirem
        self.HOME_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.DATASETS_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.IMAGES_DIR.mkdir(exist_ok=True)
        self.GALLERY_DIR.mkdir(exist_ok=True)
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.STATIC_DIR.mkdir(exist_ok=True)
        self.ANNOTATIONS_DIR.mkdir(exist_ok=True)
        self.TRAINING_DIR.mkdir(exist_ok=True)
        self.EXPORTS_DIR.mkdir(exist_ok=True)

    # Configurações do modelo YOLO
    YOLO_MODEL_PATH: Path = MODELS_DIR / "best.pt"
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

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
            directory.mkdir(parents=True, exist_ok=True)
            
    def __init__(self):
        """Inicializa as configurações e cria diretórios necessários"""
        self.create_directories()

settings = Settings()
from pathlib import Path
from typing import Optional, Dict, Any, Set
import os
import dotenv
from dataclasses import dataclass, field

# Carregar variáveis de ambiente do arquivo .env se existir
dotenv.load_dotenv(".env")

@dataclass
class Settings:
    # Configurações básicas
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MicroDetect"

    # Diretórios base
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    APP_DIR: Path = BASE_DIR / "app"

    # Diretório no home do usuário
    HOME_DIR: Path = Path.home() / ".microdetect"

    # Diretórios de dados (agora no ~/.microdetect)
    DATA_DIR: Path = HOME_DIR / "data"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    MODELS_DIR: Path = DATA_DIR / "models"
    IMAGES_DIR: Path = DATA_DIR / "images"
    GALLERY_DIR: Path = DATA_DIR / "gallery"
    TEMP_DIR: Path = DATA_DIR / "temp"
    STATIC_DIR: Path = DATA_DIR / "static"

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
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
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

    # Configurações do modelo YOLO
    YOLO_MODEL_PATH: Path = MODELS_DIR / "best.pt"
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # Configurações de processamento
    MAX_WORKERS: int = os.cpu_count() or 4
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))


settings = Settings()
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
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
    SECRET_KEY: str = "your-secret-key-here"  # Em produção, use variável de ambiente
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Configurações de upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/tiff"}
    
    # Configurações do banco de dados
    DATABASE_URL: str = f"sqlite:///{HOME_DIR}/microdetect.db"
    
    # Configurações do modelo YOLO
    YOLO_MODEL_PATH: Path = MODELS_DIR / "best.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    # Configurações de processamento
    MAX_WORKERS: int = os.cpu_count() or 4
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 
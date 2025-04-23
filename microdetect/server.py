"""
Módulo de inicialização do servidor FastAPI
"""

import os
import uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from microdetect import __version__
from microdetect.api import api_router
from microdetect.core.config import settings
from microdetect.database.database import init_database

logger = logging.getLogger(__name__)


def setup_app(data_dir=None):
    """Configura a aplicação FastAPI"""
    # Configurar diretório de dados se especificado
    if data_dir:
        # Atualizar a configuração global com o diretório especificado
        os.environ["MICRODETECT_DATA_DIR"] = data_dir
        settings.update_paths(data_dir)

    # Criar diretórios necessários
    setup_user_directories()

    # Inicializar banco de dados
    init_database()

    # Criar app FastAPI
    app = FastAPI(
        title="MicroDetect API",
        description="API para detecção de microorganismos",
        version=__version__,
    )

    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Adicionar rotas da API
    app.include_router(api_router, prefix="/api/v1")

    # Servir arquivos estáticos
    if os.path.exists(settings.IMAGES_DIR):
        app.mount("/images", StaticFiles(directory=str(settings.IMAGES_DIR)), name="images")

    # Endpoint raiz
    @app.get("/")
    async def root():
        return {
            "status": "ok",
            "message": "MicroDetect API está funcionando",
            "version": __version__
        }

    # Endpoint de saúde
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy"
        }

    return app


def setup_user_directories():
    """Configura os diretórios necessários em ~/.microdetect"""
    try:
        # Criar diretório base em ~/.microdetect
        home_dir = Path.home() / ".microdetect"
        home_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Diretório base criado: {home_dir}")

        # Criar estrutura de diretórios para dados
        data_subdirs = [
            "data",
            "data/datasets",
            "data/models",
            "data/images",
            "data/gallery",
            "data/temp",
            "data/static"
        ]

        for subdir in data_subdirs:
            dir_path = home_dir / subdir
            dir_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Diretório criado: {dir_path}")

        # Garantir que as permissões estejam corretas
        os.system(f"chmod -R 755 {home_dir}")

        return True
    except Exception as e:
        logger.error(f"Erro ao configurar diretórios do usuário: {e}")
        return False


def start_server(host="127.0.0.1", port=8000, data_dir=None):
    """Inicia o servidor FastAPI"""
    logger.info(f"Iniciando servidor na porta {port}...")
    app = setup_app(data_dir)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
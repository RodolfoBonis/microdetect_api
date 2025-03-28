from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from microdetect.api import api_router  # Import the api_router
from microdetect.core.config import settings
import os

# Criação básica da aplicação FastAPI
app = FastAPI(
    title="MicroDetect API",
    description="API para o sistema MicroDetect de análise microscópica",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adicionar as rotas da API
app.include_router(api_router, prefix="/api/v1")

# Montar diretório de imagens como rota estática
if os.path.exists(settings.IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=str(settings.IMAGES_DIR)), name="images")
    print(f"Servindo imagens estáticas de: {settings.IMAGES_DIR}")

# Endpoint raiz
@app.get("/")
async def root():
    """Endpoint raiz para verificação de saúde da API."""
    return {
        "status": "ok",
        "message": "MicroDetect API está funcionando",
        "version": "1.0.0"
    }

# Endpoint de saúde
@app.get("/health")
async def health_check():
    """Endpoint para verificação de saúde da API."""
    return {
        "status": "healthy"
    }
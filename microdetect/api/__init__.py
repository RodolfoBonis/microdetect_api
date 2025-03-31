from fastapi import APIRouter
from microdetect.api.endpoints import training, models, inference, hyperparams
from microdetect.api.endpoints import datasets, annotations, images, system

api_router = APIRouter()

# Incluir rotas dos diferentes endpoints
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(images.router, prefix="/images", tags=["images"])
api_router.include_router(annotations.router, prefix="/annotations", tags=["annotations"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(hyperparams.router, prefix="/hyperparams", tags=["hyperparams"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
api_router.include_router(system.router, prefix="/system", tags=["system"])

# API do MicroDetect

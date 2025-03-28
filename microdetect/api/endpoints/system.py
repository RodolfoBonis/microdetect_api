from fastapi import APIRouter
import platform
import os
from typing import Dict, Any
import psutil
import GPUtil
from microdetect import __version__

router = APIRouter()

def get_version_from_package() -> str:
    try:
        return __version__
    except Exception as e:
        print(f"Erro ao obter a versão do pacote: {str(e)}")
        return "unknown"

def get_gpu_info() -> Dict[str, Any]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon (M1, M2, etc.)
        gpu_info = {
            "model": "Apple Silicon GPU",
            "memory": "Shared with system memory",
            "available": True,
        }
    else:
        # Other systems
        gpus = GPUtil.getGPUs()
        gpu_info = {
            "model": gpus[0].name if gpus else "No GPU found",
            "memory": f"{gpus[0].memoryTotal}MB" if gpus else "0MB",
            "available": bool(gpus),
        }
    return gpu_info

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """
    Retorna informações sobre o status do sistema, incluindo GPU, armazenamento,
    versão do servidor e status da câmera.
    """
    try:
        # Informações de sistema operacional
        os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
        
        # Informações de CPU
        cpu_info = {
            "model": platform.processor() or "CPU Desconhecido",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "usage": f"{psutil.cpu_percent()}%"
        }
        
        # Informações de memória RAM
        memory = psutil.virtual_memory()
        memory_info = {
            "total": f"{memory.total / (1024 ** 3):.1f}GB",
            "available": f"{memory.available / (1024 ** 3):.1f}GB",
            "percentage": memory.percent
        }

        # Informações de GPU
        gpu_info = get_gpu_info()

        # Informações de armazenamento
        disk_usage = psutil.disk_usage('/')
        storage_info = {
            "used": f"{disk_usage.used / (1024 ** 3):.1f}GB",
            "total": f"{disk_usage.total / (1024 ** 3):.1f}GB",
            "percentage": disk_usage.percent,
        }

        # Informações do servidor
        server_info = {
            "version": get_version_from_package(),
            "active": True,
        }

        return {
            "os": os_info,
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "storage": storage_info,
            "server": server_info,
        }
    except Exception as e:
        # Log do erro e retorno de informações básicas
        print(f"Erro ao obter status do sistema: {str(e)}")
        return {
            "error": str(e),
            "os": "Desconhecido",
            "cpu": {"model": "Desconhecido", "cores": 0, "threads": 0, "usage": "0%"},
            "memory": {"total": "0GB", "available": "0GB", "percentage": 0},
            "gpu": {"model": "Erro ao detectar", "memory": "0", "available": False},
            "storage": {"used": "0", "total": "0", "percentage": 0},
            "server": {"version": "unknown", "active": True},
        }
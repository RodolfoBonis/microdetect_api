"""
Configuração do Metal Performance Shaders (MPS) para PyTorch no MacOS.
Este módulo deve ser importado antes de qualquer outra importação do PyTorch
para garantir que o ambiente seja configurado corretamente.
"""

import os
import sys
import platform
import multiprocessing

def configure_mps():
    """
    Configura o ambiente para usar MPS (Metal Performance Shaders) no MacOS
    de forma segura com multiprocessing.
    """
    # Verificar se é MacOS
    if platform.system() != "Darwin":
        return False
    
    try:
        # Configurar multiprocessing para usar 'spawn' em vez de 'fork'
        # Isso evita o erro de fork com MPS no MacOS
        multiprocessing.set_start_method('spawn', force=True)
        
        # Configurar o ambiente para MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Melhorar a performance e estabilidade
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Permitir que o PyTorch use MPS no MacOS
        return True
    except Exception as e:
        print(f"Erro ao configurar MPS: {e}")
        return False

def is_mps_available():
    """
    Verifica se MPS está disponível no sistema.
    """
    if platform.system() != "Darwin":
        return False
    
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except:
        pass
    
    return False

def get_device():
    """
    Retorna o dispositivo PyTorch mais apropriado (cuda, mps, ou cpu).
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"

# Configurar MPS ao importar este módulo
configure_mps() 
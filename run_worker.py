#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para iniciar os workers Celery com a configuração correta para MacOS.
Isso garante que o multiprocessing seja configurado adequadamente para
funcionar com o Metal Performance Shaders (MPS).
"""

import os
import sys
import platform
import subprocess
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Verificar se é MacOS
    is_macos = platform.system() == "Darwin"
    
    if is_macos:
        logger.info("Detectado MacOS - Configurando ambiente para MPS")
        # Configurar ambiente para MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Configurar método de multiprocessing
        os.environ["PYTHONEXECUTABLE"] = sys.executable
        os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
        
        # Forçar 'spawn' para multiprocessing
        os.environ["PYTHONSTART_METHOD"] = "spawn"
        
        # Desativar semáforos da biblioteca billiard no fork
        os.environ["BILLIARD_USE_SEMAPHORES"] = "0"
        
        logger.info("Usando processamento 'solo' para evitar problemas com fork")
        cmd = [
            sys.executable, "-m", "celery", 
            "-A", "microdetect.core.celery_app", "worker", 
            "--loglevel=info", 
            "--pool=solo"  # Importante: usar 'solo' em vez de 'prefork' no MacOS
        ]
    else:
        logger.info("Usando configurações padrão para Celery")
        cmd = [
            sys.executable, "-m", "celery", 
            "-A", "microdetect.core.celery_app", "worker", 
            "--loglevel=info"
        ]
    
    # Adicionar argumentos extras da linha de comando
    cmd.extend(sys.argv[1:])
    
    logger.info(f"Executando comando: {' '.join(cmd)}")
    
    # Executar o processo Celery
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 
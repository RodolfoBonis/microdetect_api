import os
import psutil
import time
import threading
from typing import Dict, Any, List, Optional, Callable
import logging
from microdetect.schemas.hyperparam_search import ResourceUsage

# Configurar logging
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Serviço para monitoramento de recursos do sistema (CPU, RAM, GPU)."""
    
    def __init__(self):
        self.monitoring = False
        self.monitoring_thread = None
        self._resources_history = []
        self._callback = None
        self._interval = 1.0  # segundos
        
        # Detectar suporte a GPU
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("No GPU detected, using CPU only")
        except ImportError:
            logger.info("PyTorch not available, cannot detect GPU")
    
    def start_monitoring(self, interval: float = 1.0, callback: Optional[Callable[[ResourceUsage], None]] = None):
        """
        Inicia o monitoramento de recursos em uma thread separada.
        
        Args:
            interval: Intervalo entre medições em segundos
            callback: Função a ser chamada com cada medição
        """
        if self.monitoring:
            return
            
        self.monitoring = True
        self._interval = interval
        self._callback = callback
        self._resources_history = []
        
        # Iniciar thread de monitoramento
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Resource monitoring started with {interval}s interval")
    
    def stop_monitoring(self) -> List[ResourceUsage]:
        """
        Para o monitoramento de recursos e retorna o histórico.
        
        Returns:
            Lista de medições de recursos
        """
        if not self.monitoring:
            return self._resources_history
            
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
            
        logger.info(f"Resource monitoring stopped, collected {len(self._resources_history)} samples")
        return self._resources_history
    
    def get_current_usage(self) -> ResourceUsage:
        """
        Obtém o uso atual de recursos.
        
        Returns:
            Uso atual de recursos
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        gpu_percent = None
        gpu_memory_percent = None
        
        # Obter métricas de GPU se disponível
        if self.has_gpu:
            try:
                import torch
                gpu_percent = self._get_gpu_utilization()
                gpu_memory_percent = self._get_gpu_memory_percent()
            except:
                pass
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent
        )
    
    def get_average_usage(self) -> ResourceUsage:
        """
        Calcula o uso médio de recursos durante o monitoramento.
        
        Returns:
            Uso médio de recursos
        """
        if not self._resources_history:
            return self.get_current_usage()
            
        cpu_values = [r.cpu_percent for r in self._resources_history]
        memory_values = [r.memory_percent for r in self._resources_history]
        
        # Valores de GPU podem ser None
        gpu_values = [r.gpu_percent for r in self._resources_history if r.gpu_percent is not None]
        gpu_memory_values = [r.gpu_memory_percent for r in self._resources_history if r.gpu_memory_percent is not None]
        
        return ResourceUsage(
            cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            memory_percent=sum(memory_values) / len(memory_values) if memory_values else 0,
            gpu_percent=sum(gpu_values) / len(gpu_values) if gpu_values else None,
            gpu_memory_percent=sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else None
        )
    
    def get_max_usage(self) -> ResourceUsage:
        """
        Obtém o uso máximo de recursos durante o monitoramento.
        
        Returns:
            Uso máximo de recursos
        """
        if not self._resources_history:
            return self.get_current_usage()
            
        cpu_values = [r.cpu_percent for r in self._resources_history]
        memory_values = [r.memory_percent for r in self._resources_history]
        
        # Valores de GPU podem ser None
        gpu_values = [r.gpu_percent for r in self._resources_history if r.gpu_percent is not None]
        gpu_memory_values = [r.gpu_memory_percent for r in self._resources_history if r.gpu_memory_percent is not None]
        
        return ResourceUsage(
            cpu_percent=max(cpu_values) if cpu_values else 0,
            memory_percent=max(memory_values) if memory_values else 0,
            gpu_percent=max(gpu_values) if gpu_values else None,
            gpu_memory_percent=max(gpu_memory_values) if gpu_memory_values else None
        )
    
    def _monitor_loop(self):
        """Loop principal de monitoramento em thread separada."""
        while self.monitoring:
            try:
                # Obter uso atual
                usage = self.get_current_usage()
                
                # Adicionar ao histórico
                self._resources_history.append(usage)
                
                # Chamar callback se existir
                if self._callback:
                    try:
                        self._callback(usage)
                    except Exception as e:
                        logger.error(f"Error in resource monitoring callback: {e}")
                
                # Aguardar intervalo
                time.sleep(self._interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self._interval)
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Obtém a utilização da GPU em porcentagem."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            # Tentar usar nvidia-smi via subprocess
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                return float(result.decode('utf-8').strip())
            except:
                # Fallback para valor aproximado baseado em PyTorch
                # (esta não é uma medida precisa, apenas uma aproximação)
                return 0.0  # Não é possível obter diretamente via PyTorch
        except:
            return None
    
    def _get_gpu_memory_percent(self) -> Optional[float]:
        """Obtém a percentagem de memória da GPU em uso."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
                
            # Tentar usar nvidia-smi via subprocess
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
                memory_info = result.decode('utf-8').strip().split(',')
                used = float(memory_info[0])
                total = float(memory_info[1])
                return (used / total) * 100
            except:
                # Fallback para PyTorch
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                if memory_reserved == 0:
                    return 0.0
                return (memory_allocated / memory_reserved) * 100
        except:
            return None 
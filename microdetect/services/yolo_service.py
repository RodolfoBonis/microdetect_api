import os
import logging
import time
import traceback

import torch
import platform
import math
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import random

from microdetect.core.config import settings
from microdetect.models import Dataset
from microdetect.models.model import Model
from microdetect.services.dataset_service import DatasetService
from microdetect.database.database import get_db
from microdetect.core.mps_config import is_mps_available, get_device

# Verificar disponibilidade de aceleração (CUDA ou MPS)
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = is_mps_available()
logger = logging.getLogger(__name__)
logger.info(f"CUDA available: {CUDA_AVAILABLE}")
logger.info(f"MPS available: {MPS_AVAILABLE}")

class YOLOService:
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Determinar o dispositivo a ser usado
        if settings.FORCE_CPU:
            self.device = "cpu"
        elif CUDA_AVAILABLE and settings.USE_CUDA:
            self.device = "cuda:0"
        elif MPS_AVAILABLE and settings.USE_MPS and platform.system() == "Darwin":
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logger.info(f"YOLOService usando dispositivo: {self.device}")
        
        # Carregar modelo padrão se existir
        self.model = None
        default_model_path = self.models_dir / "best.pt"
        if default_model_path.exists():
            self.load_model(str(default_model_path))
        
        self._model_cache = {}

    async def train(
        self,
        dataset_id: int,
        model_type: str,
        model_version: str,
        hyperparameters: Dict[str, Any] = None,
        callback: Any = None,
        db_session = None,
        data_yaml_path: str = None
    ) -> Dict[str, Any]:
        """
        Treina um modelo YOLO.
        
        Args:
            dataset_id: ID do dataset
            model_type: Tipo do modelo (ex: "yolov8")
            model_version: Versão do modelo
            hyperparameters: Parâmetros de treinamento
            callback: Função de callback para progresso
            db_session: Sessão do banco de dados
            data_yaml_path: Caminho para o arquivo data.yaml (opcional)
            
        Returns:
            Métricas de treinamento
        """
        try:
            # Garantir que os parâmetros são um dicionário válido
            hyperparameters = hyperparameters or {}
            
            # Verificar e ajustar o dispositivo, se necessário
            if "device" in hyperparameters and hyperparameters["device"] == "auto" and not CUDA_AVAILABLE:
                hyperparameters["device"] = "cpu"
                logger.info("CUDA não disponível. Forçando device=cpu para treinamento.")
            elif "device" not in hyperparameters and not CUDA_AVAILABLE:
                hyperparameters["device"] = "cpu"
                logger.info("CUDA não disponível. Adicionando device=cpu para treinamento.")
            
            # Se o caminho do data.yaml foi fornecido, usar diretamente
            if data_yaml_path:
                logger.info(f"Usando arquivo data.yaml fornecido: {data_yaml_path}")
                # Verificar se o arquivo existe
                if not os.path.exists(data_yaml_path):
                    logger.error(f"Arquivo data.yaml fornecido não encontrado: {data_yaml_path}")
                    raise FileNotFoundError(f"Arquivo data.yaml fornecido não encontrado: {data_yaml_path}")
            else:
                # Garantir que o dataset esteja preparado para treinamento e obter o caminho correto do data.yaml
                if db_session:
                    dataset_service = DatasetService(db_session)
                    # Obter o nome do dataset para construir o caminho correto
                    dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
                    if not dataset:
                        raise ValueError(f"Dataset {dataset_id} não encontrado")
                    
                    # Preparar o dataset e obter o caminho correto do data.yaml
                    data_yaml_path = dataset_service.prepare_for_training(dataset_id)
                    
                    # Verificar se o arquivo existe
                    if not os.path.exists(data_yaml_path):
                        logger.error(f"Arquivo data.yaml não encontrado em: {data_yaml_path}")
                        raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_yaml_path}")
                    
                    logger.info(f"Usando arquivo data.yaml em: {data_yaml_path}")
                else:
                    # Se não tiver sessão do banco, usar o caminho correto com base no nome do dataset
                    # Isso requer que tenhamos o nome do dataset, então vamos tentar obtê-lo
                    try:
                        # Criar temporariamente uma sessão do banco para obter o nome do dataset
                        temp_db = next(get_db())
                        dataset = temp_db.query(Dataset).filter(Dataset.id == dataset_id).first()
                        if dataset:
                            # Construir o caminho correto
                            data_yaml_path = str(settings.TRAINING_DIR / dataset.name / "data.yaml")
                            temp_db.close()
                        else:
                            raise ValueError(f"Dataset {dataset_id} não encontrado")
                    except Exception as e:
                        logger.error(f"Erro ao obter o nome do dataset: {str(e)}")
                        # Fallback para o caminho padrão antigo, que provavelmente não existirá
                        data_yaml_path = f"data/datasets/{dataset_id}/data.yaml"
            
            # Registrar o caminho do data.yaml
            logger.info(f"Usando arquivo data.yaml: {data_yaml_path}")
            
            # Verificar se o arquivo existe
            if not os.path.exists(data_yaml_path):
                logger.error(f"Arquivo data.yaml não encontrado em: {data_yaml_path}")
                raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_yaml_path}")
                
            # Configurar parâmetros padrão
            params = {
                "epochs": 100,
                "batch": 16,  # YOLO usa 'batch', não 'batch_size'
                "imgsz": 640,
                "device": "auto",
                "workers": 8,
                "project": str(self.models_dir),
                "name": f"dataset_{dataset_id}",
                "exist_ok": True,
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "seed": 0,
                "deterministic": True,
                "single_cls": False,
                "rect": False,
                "cos_lr": False,
                "close_mosaic": 0,
                "resume": False,
                "amp": True,
                "fraction": 1.0,
                "cache": False,
                "overlap_mask": True,
                "mask_ratio": 4,
                "dropout": 0.0,
                "val": True,
                "save": True,
                "save_json": False,
                "save_hybrid": False,
                "conf": 0.001,
                "iou": 0.6,
                "max_det": 300,
                "half": False,
                "dnn": False,
                "plots": True,
            }
            
            # Registrar o tipo do modelo e hiperparâmetros recebidos para debug
            print(f"Treinando modelo {model_type}{model_version} com hiperparâmetros: {hyperparameters}")
            
            # Atualizar com parâmetros fornecidos
            if hyperparameters:
                # Processar parâmetros específicos do search_space
                if "model_type" in hyperparameters:
                    model_type = hyperparameters["model_type"].lower()  # Convertendo para minúsculas
                    logger.info(f"Tipo de modelo selecionado: {model_type}")
                
                if "model_size" in hyperparameters:
                    model_version = hyperparameters["model_size"]
                    logger.info(f"Tamanho do modelo selecionado: {model_version}")
                
                if "imgsz" in hyperparameters:
                    params["imgsz"] = hyperparameters["imgsz"]
                    logger.info(f"Tamanho da imagem selecionado: {params['imgsz']}")
                
                if "optimizer" in hyperparameters:
                    params["optimizer"] = hyperparameters["optimizer"]
                    logger.info(f"Otimizador selecionado: {params['optimizer']}")
                
                if "device" in hyperparameters:
                    device = hyperparameters["device"]
                    if device == "auto":
                        # Usar CUDA se disponível, caso contrário CPU
                        params["device"] = "0" if settings.USE_CUDA else "cpu"
                    elif device.startswith("GPU"):
                        # Extrair o número da GPU se fornecido
                        gpu_num = device.replace("GPU", "").strip()
                        params["device"] = gpu_num if gpu_num else "0"
                    else:
                        params["device"] = device
                    logger.info(f"Dispositivo selecionado: {params['device']}")
                
                # Verificar e converter parâmetros incompatíveis
                if "batch_size" in hyperparameters:
                    hyperparameters["batch"] = hyperparameters.pop("batch_size")
                
                # Converter learning_rate para lr0 (nome correto para YOLO)
                if "learning_rate" in hyperparameters:
                    value = hyperparameters.pop("learning_rate")
                    # Se for um dicionário com min/max, usar um valor médio
                    if isinstance(value, dict) and "min" in value and "max" in value:
                        hyperparameters["lr0"] = (value["min"] + value["max"]) / 2
                    else:
                        hyperparameters["lr0"] = value
                
                # Processar outros parâmetros que podem vir como min/max
                for param in ["epochs", "batch"]:
                    if param in hyperparameters and isinstance(hyperparameters[param], dict):
                        if "min" in hyperparameters[param] and "max" in hyperparameters[param]:
                            # Para inteiros, pegar um valor médio arredondado
                            if param == "epochs":
                                min_val = hyperparameters[param]["min"]
                                max_val = hyperparameters[param]["max"]
                                hyperparameters[param] = round((min_val + max_val) / 2)
                            elif param == "batch":
                                min_val = hyperparameters[param]["min"]
                                max_val = hyperparameters[param]["max"]
                                # Prefira potências de 2 para batch
                                hyperparameters[param] = 2 ** round(math.log2((min_val + max_val) / 2))
                
                # Verificar o parâmetro epochs explicitamente
                if "epochs" in hyperparameters:
                    try:
                        # Garantir que epochs seja um inteiro
                        epochs_value = int(hyperparameters["epochs"])
                        hyperparameters["epochs"] = epochs_value
                    except (ValueError, TypeError) as e:
                        print(f"Erro ao converter 'epochs' para inteiro: {str(e)}")
                        # Usar valor padrão se falhar
                        hyperparameters.pop("epochs", None)
                    
                # Remover parâmetros inválidos para evitar erros
                invalid_params = ["model_type", "model_size", "model_version"]
                for param in invalid_params:
                    if param in hyperparameters:
                        hyperparameters.pop(param)
                        
                # Atualizar com os parâmetros corrigidos
                params.update(hyperparameters)
            
            # Carregar modelo base
            try:
                model = YOLO(f"{model_type}{model_version}.pt")
            except Exception as e:
                print(f"Erro ao carregar modelo base: {str(e)}")
                raise
            
            # Definir uma função para monitorar o progresso, se callback for fornecido
            if callback:
                try:
                    # Verificar tipo do callback
                    if not callable(callback):
                        print(f"AVISO: Callback não é uma função chamável: {type(callback)}")
                        # Criar uma função wrapper para segurança
                        original_callback = callback
                        callback = lambda x: print(f"Callback wrapper: não foi possível chamar {original_callback}")
                    
                    # Configurar um callback para o YOLO
                    class ProgressCallback:
                        def __init__(self, callback, total_epochs, update_interval=0.1):
                            self.callback = callback
                            self.total_epochs = total_epochs
                            self.update_interval = update_interval
                            self.last_update_time = time.time()
                            self.current_epoch = 0
                            self.validation_results = {}
                            self.epoch_start_time = None
                            self.epoch_times = []
                            
                            # Imprimir informação para depuração
                            print(f"ProgressCallback inicializado com callback tipo: {type(callback)}")
                        
                        def on_train_batch_end(self, trainer):
                            """Callback chamado após cada batch de treinamento - utilizado para atualizações frequentes"""
                            current_time = time.time()
                            
                            # Log para depuração
                            logger.debug(f"on_train_batch_end chamado: epoch={trainer.epoch}")
                            
                            # Limitar a frequência de atualizações para não sobrecarregar
                            if current_time - self.last_update_time < self.update_interval:
                                return
                                
                            # Atualizar timestamp
                            self.last_update_time = current_time
                            
                            # Extrair métricas básicas do batch atual
                            # Obter informações seguras do trainer
                            batch_info = {}
                            # Tentar obter informações do batch de diferentes formas
                            if hasattr(trainer, 'batch_idx'):
                                batch_info["current_batch"] = trainer.batch_idx
                            elif hasattr(trainer, 'step'):
                                batch_info["current_batch"] = trainer.step
                            
                            if hasattr(trainer, 'num_batches'):
                                batch_info["total_batches"] = trainer.num_batches
                            elif hasattr(trainer, 'dataloader'):
                                batch_info["total_batches"] = len(trainer.dataloader)
                                
                            current_metrics = {
                                "epoch": trainer.epoch,
                                "batch_size": getattr(trainer, 'batch_size', 0),
                                "total_epochs": trainer.epochs,
                                "loss": float(trainer.loss.detach().cpu().numpy() if hasattr(trainer, 'loss') else 0.0),
                                "progress_type": "batch",  # Indicar que é uma atualização de batch
                                **batch_info  # Incluir informações do batch obtidas dinamicamente
                            }
                            
                            # Log antes de chamar o callback
                            logger.info(f"Enviando atualização de batch: epoch={trainer.epoch}, batch={batch_info.get('current_batch', 'N/A')}")
                            
                            # Chamar o callback com as métricas básicas
                            if callable(self.callback):
                                try:
                                    self.callback(current_metrics)
                                except Exception as e:
                                    print(f"Erro ao chamar callback: {str(e)}")
                            else:
                                print(f"AVISO: self.callback não é uma função chamável: {type(self.callback)}")
                        
                        def on_train_epoch_start(self, trainer):
                            """Callback chamado no início de cada época de treinamento."""
                            self.current_epoch = trainer.epoch
                            self.epoch_start_time = time.time()
                        
                        def on_train_epoch_end(self, trainer):
                            """Callback chamado após cada época de treinamento."""
                            current_time = time.time()
                            
                            # Calcular o tempo da época atual
                            if self.epoch_start_time is not None:
                                elapsed = current_time - self.epoch_start_time
                                self.epoch_times.append(elapsed)
                            
                            if current_time - self.last_update_time >= self.update_interval:
                                self.last_update_time = current_time
                                
                                # Atualizar contador de época
                                self.current_epoch = trainer.epoch
                                
                                # Obter as métricas atuais
                                metrics = trainer.metrics
                                
                                # Calcular ETA com segurança
                                eta_seconds = 0
                                if hasattr(trainer, 'epoch_time') and trainer.epoch_time is not None and hasattr(trainer.epoch_time, 'avg'):
                                    eta_seconds = trainer.epoch_time.avg * (trainer.epochs - trainer.epoch)
                                
                                # Adicionar época atual
                                epoch_metrics = {
                                    "epoch": self.current_epoch,
                                    "total_epochs": self.total_epochs,
                                    "loss": float(metrics.get("train/box_loss", 0) + metrics.get("train/cls_loss", 0)),
                                    "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
                                    "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
                                    "precision": float(metrics.get("metrics/precision(B)", 0.0)),
                                    "recall": float(metrics.get("metrics/recall(B)", 0.0)),
                                    "val_loss": float(metrics.get("val/box_loss", 0.0)),
                                    "eta_seconds": eta_seconds,
                                    "progress_type": "epoch"
                                }
                                
                                # Adicionar métricas de validação anteriores, se disponíveis
                                if self.validation_results:
                                    for k, v in self.validation_results.items():
                                        if k not in epoch_metrics and k != "progress_type":
                                            epoch_metrics[k] = v
                                
                                # Calcular tempo médio por época e ETA
                                avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
                                remaining_epochs = self.total_epochs - self.current_epoch
                                eta_seconds = int(remaining_epochs * avg_epoch_time)
                                
                                # Adicionar ETA estimado de forma segura
                                epoch_metrics["eta_seconds"] = eta_seconds
                                epoch_metrics["avg_epoch_time"] = avg_epoch_time
                                
                                # Verificar se callback é uma função antes de chamar
                                if callable(self.callback):
                                    try:
                                        self.callback(epoch_metrics)
                                    except Exception as e:
                                        print(f"Erro ao chamar callback: {str(e)}")
                                else:
                                    print(f"AVISO: self.callback não é uma função chamável: {type(self.callback)}")
                        
                        def on_val_end(self, validator):
                            """Callback chamado após validação."""
                            print("VALIDAÇÃO CONCLUÍDA!")
                            
                            # Verificar se validator tem métricas
                            if not hasattr(validator, 'metrics') or validator.metrics is None:
                                print("AVISO: validator não tem métricas")
                                return
                            
                            # Acesso seguro às métricas do validator
                            try:
                                metrics = validator.metrics
                                
                                # DEBUG: Imprimir métricas de validação
                                
                                # Construir relatório de métricas
                                val_epoch_metrics = {
                                    "epoch": self.current_epoch,
                                    "total_epochs": self.total_epochs,
                                    "progress_type": "validation",
                                    "map50": float(metrics.results_dict.get('metrics/mAP50(B)', 0.0)),
                                    "map50_95": float(metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                                    "precision": float(metrics.results_dict.get('metrics/precision(B)', 0.0)),
                                    "recall": float(metrics.results_dict.get('metrics/recall(B)', 0.0)),
                                    "fitness": float(metrics.results_dict.get('fitness', 0.0))
                                }
                                
                                # Debug métricas extraídas
                                print(f"Métricas extraídas: {val_epoch_metrics}")
                                
                                # Verificar se callback é uma função antes de chamar
                                if callable(self.callback):
                                    try:
                                        self.callback(val_epoch_metrics)
                                    except Exception as e:
                                        print(f"Erro ao chamar callback: {str(e)}")
                                else:
                                    print(f"AVISO: self.callback não é uma função chamável: {type(self.callback)}")
                                    
                            except Exception as e:
                                print(f"Erro ao processar métricas de validação: {str(e)}")
                                print(f"Traceback: {traceback.format_exc()}")

                    # Registrar os callbacks
                    progress_callback = ProgressCallback(callback, params["epochs"])

                    # Verificar se callback é válido
                    if not callable(callback):
                        print(f"AVISO: O callback fornecido não é uma função: {type(callback)}")
                        # Corrigir para evitar o erro
                        progress_callback.callback = lambda x: print(f"Callback inválido: {x}")

                    # Adicionar os callbacks
                    model.add_callback("on_train_batch_end", progress_callback.on_train_batch_end)
                    model.add_callback("on_train_epoch_start", progress_callback.on_train_epoch_start)
                    model.add_callback("on_train_epoch_end", progress_callback.on_train_epoch_end)
                    model.add_callback("on_val_end", progress_callback.on_val_end)
                
                except Exception as e:
                    print(f"Erro ao configurar callbacks: {str(e)}")
                    # Continuar o treinamento sem callbacks se houver erro
            
            # Registrar os parâmetros finais para debug
            print(f"Parâmetros finais de treinamento: {params}")
            
            # Em yolo_service.py, ao configurar params para model.train()
            params.update({
                "val": True,  # Garantir que validação esteja ativada
            })
            
            # Preparar os hiperparâmetros para o treinamento
            processed_params = {}

            # Converter o formato de hiperparâmetros para o formato que o YOLOv8 espera
            for param_name, param_value in hyperparameters.items():
                # Verificar se o parâmetro é um intervalo (min/max)
                if isinstance(param_value, dict) and "min" in param_value and "max" in param_value:
                    # Processar parâmetros específicos
                    if param_name == "learning_rate":
                        # Para learning_rate, mapear para lr0 no YOLOv8
                        processed_params["lr0"] = random.uniform(param_value["min"], param_value["max"])
                    elif param_name == "batch_size":
                        # Para batch_size, mapear para batch no YOLOv8, garantindo inteiro
                        processed_params["batch"] = int(random.uniform(param_value["min"], param_value["max"]))
                    elif param_name == "epochs":
                        # Para epochs, usar diretamente
                        processed_params["epochs"] = int(random.uniform(param_value["min"], param_value["max"]))
                    else:
                        # Para outros parâmetros com intervalo, escolher um valor aleatório
                        if isinstance(param_value["min"], int) and isinstance(param_value["max"], int):
                            processed_params[param_name] = random.randint(param_value["min"], param_value["max"])
                        else:
                            processed_params[param_name] = random.uniform(param_value["min"], param_value["max"])
                else:
                    # Se não for um intervalo, usar o valor diretamente
                    processed_params[param_name] = param_value

            # Usar os parâmetros processados para treinamento
            # Treinar modelo
            results = model.train(
                data=data_yaml_path,
                **params
            )
            
            # Extrair métricas
            metrics = {}
            
            # Verificar se results_dict existe e extrair métricas com segurança
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                
                # Extrair métricas seguramente, com valores padrão se não existirem
                metrics = {
                    "epochs": results_dict.get("epochs", params.get("epochs", 0)),
                    "best_epoch": results_dict.get("best_epoch", 0),
                    "best_map50": results_dict.get("best_map50", 0.0),
                    "best_map": results_dict.get("best_map", 0.0),
                    "final_map50": results_dict.get("final_map50", 0.0),
                    "final_map": results_dict.get("final_map", 0.0),
                    "train_time": results_dict.get("train_time", 0.0),
                    "val_time": results_dict.get("val_time", 0.0),
                    # Adicionando novas métricas
                    "precision": results_dict.get("metrics/precision(B)", 0.0),
                    "recall": results_dict.get("metrics/recall(B)", 0.0),
                    "f1_score": results_dict.get("metrics/f1(B)", 0.0),
                    "best_precision": results_dict.get("best_precision", 0.0),
                    "best_recall": results_dict.get("best_recall", 0.0),
                    "best_f1_score": results_dict.get("best_f1", 0.0)
                }
            else:
                # Se results_dict não existir, usar valores padrão
                logger.warning("Objeto results não contém results_dict. Usando valores padrão para métricas.")
                metrics = {
                    "epochs": params.get("epochs", 0),
                    "best_epoch": 0,
                    "best_map50": 0.0, 
                    "best_map": 0.0,
                    "final_map50": 0.0,
                    "final_map": 0.0,
                    "train_time": 0.0,
                    "val_time": 0.0,
                    # Adicionando novas métricas com valores padrão
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "best_precision": 0.0,
                    "best_recall": 0.0,
                    "best_f1_score": 0.0
                }
            
            # Registrar métricas obtidas
            logger.info(f"Métricas de treinamento obtidas: {metrics}")
            
            # Registrar os resultados
            trial_result = {
                "params": processed_params,
                "metrics": {
                    **metrics,  # Incluir todas as métricas existentes
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0)
                }
            }
            
            return trial_result
            
        except Exception as e:
            print(f"Erro durante treinamento: {str(e)}")
            # Repassar a exceção após o log
            raise

    async def predict(
        self,
        model_id: int,
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Realiza inferência em uma imagem.
        
        Args:
            model_id: ID do modelo
            image_path: Caminho da imagem
            confidence_threshold: Limiar de confiança
            
        Returns:
            Tuple com lista de detecções e métricas
        """
        # Carregar modelo do cache ou do banco
        if model_id not in self._model_cache:
            model = Model.query.get(model_id)
            if not model:
                raise ValueError(f"Modelo {model_id} não encontrado")
            
            self._model_cache[model_id] = YOLO(model.filepath)
        
        # Realizar inferência
        results = self._model_cache[model_id].predict(
            source=image_path,
            conf=confidence_threshold,
            verbose=False
        )
        
        # Processar resultados
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                prediction = {
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
                predictions.append(prediction)
        
        # Extrair métricas
        metrics = {
            "inference_time": results[0].speed["inference"] / 1000,  # em segundos
            "fps": 1000 / results[0].speed["inference"],
            "num_detections": len(predictions),
        }
        
        return predictions, metrics

    async def validate(
        self,
        model_id: int,
        dataset_id: int
    ) -> Dict[str, Any]:
        """
        Valida um modelo em um dataset.
        
        Args:
            model_id: ID do modelo
            dataset_id: ID do dataset
            
        Returns:
            Métricas de validação
        """
        # Carregar modelo
        if model_id not in self._model_cache:
            model = Model.query.get(model_id)
            if not model:
                raise ValueError(f"Modelo {model_id} não encontrado")
            
            self._model_cache[model_id] = YOLO(model.filepath)
        
        # Obter o caminho correto do data.yaml
        try:
            # Criar temporariamente uma sessão do banco para obter o nome do dataset
            temp_db = next(get_db())
            dataset = temp_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                # Construir o caminho correto
                data_yaml_path = str(settings.TRAINING_DIR / dataset.name / "data.yaml")
                # Opcionalmente preparar o dataset, se necessário
                dataset_service = DatasetService(temp_db)
                dataset_service.prepare_for_training(dataset_id)
            else:
                raise ValueError(f"Dataset {dataset_id} não encontrado")
            
            # Verificar se o arquivo existe
            if not os.path.exists(data_yaml_path):
                logger.error(f"Arquivo data.yaml não encontrado em: {data_yaml_path}")
                raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_yaml_path}")
                
            logger.info(f"Usando arquivo data.yaml para validação: {data_yaml_path}")
            
        except Exception as e:
            logger.error(f"Erro ao obter o caminho do data.yaml: {str(e)}")
            # Fallback para o caminho padrão antigo
            data_yaml_path = f"data/datasets/{dataset_id}/data.yaml"
        finally:
            # Fechar a sessão temporária
            if 'temp_db' in locals():
                temp_db.close()
        
        # Validar modelo
        results = self._model_cache[model_id].val(
            data=data_yaml_path,
            verbose=True
        )
        
        # Extrair métricas
        metrics = {
            "map50": results.box.map50,
            "map": results.box.map,
            "precision": results.box.precision,
            "recall": results.box.recall,
            "f1": results.box.f1,
            "confusion_matrix": results.confusion_matrix.matrix.tolist(),
        }
        
        return metrics

    async def export(
        self,
        model_id: int,
        format: str = "onnx"
    ) -> str:
        """
        Exporta um modelo para outro formato.
        
        Args:
            model_id: ID do modelo
            format: Formato de exportação (onnx, torchscript, etc.)
            
        Returns:
            Caminho do modelo exportado
        """
        # Carregar modelo
        if model_id not in self._model_cache:
            model = Model.query.get(model_id)
            if not model:
                raise ValueError(f"Modelo {model_id} não encontrado")
            
            self._model_cache[model_id] = YOLO(model.filepath)
        
        # Exportar modelo
        export_path = self._model_cache[model_id].export(format=format)
        return export_path
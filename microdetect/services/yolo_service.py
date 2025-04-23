import os
import logging
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from microdetect.core.config import settings
from microdetect.models import Dataset
from microdetect.models.model import Model
from microdetect.services.dataset_service import DatasetService
from microdetect.database.database import get_db

# Verificar se CUDA está disponível
CUDA_AVAILABLE = torch.cuda.is_available()
logger = logging.getLogger(__name__)
logger.info(f"CUDA available in YOLOService: {CUDA_AVAILABLE}")

class YOLOService:
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda:0" if settings.USE_CUDA else "cpu"
        logger.info(f"CUDA available in YOLOService: {settings.USE_CUDA}")
        
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
                # Verificar e converter parâmetros incompatíveis
                if "batch_size" in hyperparameters:
                    hyperparameters["batch"] = hyperparameters.pop("batch_size")
                    
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
                # Configurar um callback para o YOLO
                class ProgressCallback:
                    def __init__(self):
                        self.current_epoch = 0
                        self.total_epochs = params["epochs"]
                    
                    def on_train_epoch_end(self, trainer):
                        # Extrair métricas da época atual
                        current_metrics = {
                            "epoch": trainer.epoch,
                            "total_epochs": trainer.epochs,
                            "loss": float(trainer.loss.detach().cpu().numpy() if hasattr(trainer, 'loss') else 0.0),
                            "map50": float(trainer.metrics.get("metrics/mAP50(B)", 0.0)),
                            "map": float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0)),
                            "precision": float(trainer.metrics.get("metrics/precision(B)", 0.0)),
                            "recall": float(trainer.metrics.get("metrics/recall(B)", 0.0)),
                            "val_loss": float(trainer.metrics.get("val/box_loss", 0.0)),
                            "eta_seconds": trainer.epoch_time.avg * (trainer.epochs - trainer.epoch)
                        }
                        
                        # Chamar o callback com as métricas
                        if callback:
                            callback(current_metrics)
                
                # Registrar o callback
                progress_callback = ProgressCallback()
                model.add_callback("on_train_epoch_end", progress_callback.on_train_epoch_end)
            
            # Registrar os parâmetros finais para debug
            print(f"Parâmetros finais de treinamento: {params}")
            
            # Treinar modelo
            results = model.train(
                data=data_yaml_path,
                **params
            )
            
            # Extrair métricas
            metrics = {
                "epochs": results.results_dict["epochs"],
                "best_epoch": results.results_dict["best_epoch"],
                "best_map50": results.results_dict["best_map50"],
                "best_map": results.results_dict["best_map"],
                "final_map50": results.results_dict["final_map50"],
                "final_map": results.results_dict["final_map"],
                "train_time": results.results_dict["train_time"],
                "val_time": results.results_dict["val_time"],
            }
            
            return metrics
            
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
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
from microdetect.core.config import settings
from microdetect.models.model import Model


class YOLOService:
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache = {}

    async def train(
        self,
        dataset_id: int,
        model_type: str,
        model_version: str,
        hyperparameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Treina um modelo YOLO.
        
        Args:
            dataset_id: ID do dataset
            model_type: Tipo do modelo (ex: "yolov8")
            model_version: Versão do modelo
            hyperparameters: Parâmetros de treinamento
            
        Returns:
            Métricas de treinamento
        """
        # Configurar parâmetros padrão
        params = {
            "epochs": 100,
            "batch_size": 16,
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
        
        # Atualizar com parâmetros fornecidos
        if hyperparameters:
            params.update(hyperparameters)
        
        # Carregar modelo base
        model = YOLO(f"{model_type}{model_version}.pt")
        
        # Treinar modelo
        results = model.train(
            data=f"data/datasets/{dataset_id}/data.yaml",
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
        
        # Validar modelo
        results = self._model_cache[model_id].val(
            data=f"data/datasets/{dataset_id}/data.yaml",
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
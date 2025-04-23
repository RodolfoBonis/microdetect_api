# Importar todos os modelos para que sejam registrados pelo SQLAlchemy
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset
from microdetect.models.annotation import Annotation
from microdetect.models.training_session import TrainingSession, TrainingStatus
from microdetect.models.model import Model
from microdetect.models.dataset_image import DatasetImage
from microdetect.models.inference_result import InferenceResult
from microdetect.models.hyperparam_search import HyperparamSearch, HyperparamSearchStatus
from microdetect.models.training_report import TrainingReport
from microdetect.models.base import Base, BaseModel

__all__ = [
    'Base',
    'BaseModel',
    'Dataset',
    'DatasetImage',
    'HyperparamSearch',
    'Model',
    'TrainingSession',
    'TrainingReport',
    'Annotation',
    'InferenceResult',
    'Image'
]

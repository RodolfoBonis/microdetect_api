from celery import Celery
from microdetect.core.config import settings

# Configurar Celery
celery_app = Celery(
    'microdetect',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        'microdetect.tasks.training_tasks',
        'microdetect.tasks.hyperparam_tasks'
    ]
)

# Configurações do Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Sao_Paulo',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=86400,  # 24 horas
    worker_max_tasks_per_child=1,  # Reiniciar worker após cada tarefa
    worker_prefetch_multiplier=1,  # Processar uma tarefa por vez
    task_routes={
        'microdetect.tasks.training_tasks.*': {'queue': 'training'},
        'microdetect.tasks.hyperparam_tasks.*': {'queue': 'hyperparam'}
    }
) 
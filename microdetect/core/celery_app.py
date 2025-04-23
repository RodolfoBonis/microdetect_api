from celery import Celery
from microdetect.core.config import settings

# Configurar Celery
celery_app = Celery(
    'microdetect',
    broker='amqp://microdetect:microdetect123@localhost:5672//',  # RabbitMQ como broker
    backend='redis://localhost:6379/0',  # Redis como backend
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
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=18000,  # 5 horas
    worker_max_tasks_per_child=1,  # Reinicia worker após cada task
    worker_prefetch_multiplier=1,  # Processa uma task por vez
    broker_connection_timeout=30,  # Timeout de conexão com o broker
    broker_heartbeat=10,  # Heartbeat para manter a conexão viva
    task_routes={
        'microdetect.tasks.training_tasks.*': {'queue': 'training'},
        'microdetect.tasks.hyperparam_tasks.*': {'queue': 'hyperparam'}
    }
)

# Configurar logging
celery_app.conf.update(
    worker_redirect_stdouts=False,
    worker_redirect_stdouts_level="INFO",
) 
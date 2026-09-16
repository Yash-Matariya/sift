"""
The Celery app: the background job runner for slow work (document ingestion).

Ingestion — chunking and embedding a whole document — is too slow to do inside
the upload request, so the endpoint queues it here and a separate worker process
does it. Redis is the broker (the queue) and the result backend.

`celery_eager` runs tasks in-process instead of via a worker, with no broker
needed — used in tests so the whole flow can be exercised without Redis.
"""
from celery import Celery

from app.config import settings

celery = Celery(
    "sift",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery.conf.update(
    task_always_eager=settings.celery_eager,   # tests set this True
    task_eager_propagates=True,                 # surface task errors in eager mode
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# Import tasks so they're registered with this app.
from app import tasks  # noqa: E402,F401

"""
RAG Medan v3 - Shared Service Bootstrap

Deduplikasi setup antar service:
- `create_qdrant_client` / `ensure_payload_index` / `backfill_is_active`
  (sebelumnya duplikat di rag_document & rag_web)
- `LazyModel`: holder model embedding lazy + idle unload, dengan gate
  `USE_SHARED_EMBEDDING` agar model lokal TIDAK dimuat saat shared
  embedding service aktif (hemat RAM besar).
"""
import asyncio
import gc
import logging
import time
from typing import Any, Callable, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

from config import config

logger = logging.getLogger("shared.bootstrap")


def create_qdrant_client() -> AsyncQdrantClient:
    """Create AsyncQdrantClient with memory-conscious settings (HTTP only)."""
    api_key = config.QDRANT_API_KEY or None
    return AsyncQdrantClient(
        url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}",
        api_key=api_key,
        prefer_grpc=False,
        timeout=60,
    )


async def ensure_payload_index(
    client: AsyncQdrantClient,
    collection_name: str,
    field_name: str,
    field_schema,
) -> None:
    """Best-effort payload index creation (idempotent)."""
    try:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    except Exception as exc:
        logger.debug(
            f"Skip/create payload index failed for {collection_name}.{field_name}: {exc}"
        )


async def backfill_is_active(
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """Backfill missing is_active payload for legacy points."""
    try:
        offset = None
        updated_active = 0
        updated_inactive = 0
        while True:
            points, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            active_ids = []
            inactive_ids = []
            for point in points:
                payload = dict(point.payload or {})
                if "is_active" in payload:
                    continue
                if payload.get("is_deleted", False):
                    inactive_ids.append(point.id)
                else:
                    active_ids.append(point.id)

            if active_ids:
                await client.set_payload(
                    collection_name=collection_name,
                    payload={"is_active": True},
                    points=active_ids,
                )
                updated_active += len(active_ids)
            if inactive_ids:
                await client.set_payload(
                    collection_name=collection_name,
                    payload={"is_active": False},
                    points=inactive_ids,
                )
                updated_inactive += len(inactive_ids)

            if next_offset is None:
                break
            offset = next_offset

        if updated_active or updated_inactive:
            logger.info(
                f"Backfilled is_active on {collection_name}: "
                f"active={updated_active}, inactive={updated_inactive}"
            )
    except Exception as exc:
        logger.warning(f"Backfill is_active skipped for {collection_name}: {exc}")


class LazyModel:
    """
    Lazy, gated, idle-unloading embedding model holder.

    Gate: saat `USE_SHARED_EMBEDDING=True` (default), `get()` langsung
    mengembalikan None dan model lokal TIDAK pernah dimuat — encoding
    didelegasikan ke embedding microservice via `encode_texts`.
    Saat `USE_SHARED_EMBEDDING=False`, model lokal dimuat lazy dengan
    proteksi thundering herd dan di-unload setelah idle timeout.

    `on_load` dipanggil setelah model pertama kali dimuat (untuk wiring
    instance ke search/sync module).
    """

    def __init__(
        self,
        model_path: str,
        *,
        on_load: Optional[Callable[[Any], None]] = None,
        name: str = "embedding",
    ) -> None:
        self._model_path = model_path
        self._on_load = on_load
        self._name = name
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._last_used: float = 0.0
        self._unload_task: Optional[asyncio.Task] = None

    @property
    def loaded(self) -> bool:
        """True jika model lokal sedang dimuat di memori."""
        return self._model is not None

    @property
    def model(self) -> Any:
        return self._model

    async def get(self) -> Any:
        """Return local model, atau None jika shared embedding aktif."""
        if config.USE_SHARED_EMBEDDING:
            return None

        if self._model is not None:
            self._last_used = time.time()
            return self._model

        async with self._lock:
            if self._model is not None:
                self._last_used = time.time()
                return self._model

            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading {self._name} model (lazy, local mode)...")
            self._model = SentenceTransformer(self._model_path)
            self._last_used = time.time()
            logger.info(f"Model loaded: {self._model_path}")
            if self._on_load is not None:
                self._on_load(self._model)
            return self._model

    def start_idle_unload(self) -> None:
        """Start background idle-unload loop (dipanggil di lifespan)."""
        if self._unload_task is None:
            self._unload_task = asyncio.create_task(self._idle_unload_loop())

    async def stop_idle_unload(self) -> None:
        if self._unload_task is not None:
            self._unload_task.cancel()
            self._unload_task = None

    async def _idle_unload_loop(self) -> None:
        while True:
            await asyncio.sleep(300)  # cek setiap 5 menit
            if self._model is not None and self._last_used > 0:
                idle_seconds = time.time() - self._last_used
                if idle_seconds > config.MODEL_IDLE_TIMEOUT:
                    async with self._lock:
                        if self._model is not None and (
                            time.time() - self._last_used
                        ) > config.MODEL_IDLE_TIMEOUT:
                            logger.info(
                                f"Model idle for {idle_seconds:.0f}s > "
                                f"{config.MODEL_IDLE_TIMEOUT}s, unloading..."
                            )
                            del self._model
                            self._model = None
                            gc.collect()
                            logger.info("Model unloaded, RAM freed")


def ensure_collection_params(
    dimension: int,
) -> VectorParams:
    """Standard COSINE vector params for collection creation."""
    return VectorParams(size=dimension, distance=Distance.COSINE)

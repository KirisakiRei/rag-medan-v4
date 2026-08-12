"""
RAG Medan v4 - LightRAG Adapter — Health Check Logic.

Menyediakan comprehensive health check untuk adapter service
dan dependency-nya (LightRAG Server).
"""
import logging

from services.lightrag_adapter.client import lightrag_client

logger = logging.getLogger("lightrag_adapter.health")


async def check_health() -> dict:
    """
    Comprehensive health check untuk LightRAG Adapter.

    Checks:
    - LightRAG Server reachability
    - Circuit breaker state

    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "service": "lightrag_adapter",
            "components": {
                "lightrag_server": bool,
                "circuit_breaker": "closed" | "open" | "half_open"
            }
        }
    """
    lightrag_healthy = await lightrag_client.check_health()
    circuit_state = lightrag_client.circuit_state

    if lightrag_healthy:
        status = "healthy"
    elif circuit_state == "open":
        status = "degraded"    # Circuit open, tapi service mungkin recover
    else:
        status = "unhealthy"

    return {
        "status": status,
        "service": "lightrag_adapter",
        "components": {
            "lightrag_server": lightrag_healthy,
            "circuit_breaker": circuit_state,
        },
    }

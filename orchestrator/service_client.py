"""Service client — HTTP communication with internal microservices."""
import asyncio
import logging
from typing import Dict, Any, Tuple

import httpx

logger = logging.getLogger("orchestrator")

# Global HTTP client (to be initialized by main app)
http_client: httpx.AsyncClient = None


def set_client(client: httpx.AsyncClient):
    """Set the HTTP client from main app."""
    global http_client
    http_client = client


def create_optimized_client() -> httpx.AsyncClient:
    """Create HTTP client with optimized settings for lower memory."""
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=10,           # Reduced from default 100
            max_keepalive_connections=5   # Reduced from default 20
        ),
        timeout=60.0
    )


async def call_service(
    service_url: str, 
    endpoint: str, 
    method: str = "POST", 
    data: dict = None,
    timeout: float = 120.0
) -> dict:
    """Call internal service endpoint."""
    global http_client
    url = f"{service_url}{endpoint}"
    method = method.upper()
    
    try:
        if method == "POST":
            response = await http_client.post(url, json=data, timeout=timeout)
        elif method == "PUT":
            response = await http_client.put(url, json=data, timeout=timeout)
        elif method == "GET":
            response = await http_client.get(url, timeout=timeout)
        elif method == "DELETE":
            response = await http_client.request("DELETE", url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        try:
            return response.json()
        except ValueError:
            body_preview = (response.text or "").strip()
            logger.error(
                f"[SERVICE] Non-JSON response from {url} "
                f"(status={response.status_code}): {body_preview[:300]}"
            )
            return {
                "status": "error",
                "error": f"Invalid response from service (HTTP {response.status_code})",
                "raw_response": body_preview[:300],
            }
        
    except httpx.TimeoutException:
        logger.error(f"[SERVICE] Timeout calling {url}")
        return {"status": "error", "error": "Service timeout"}
    except httpx.ConnectError:
        logger.error(f"[SERVICE] Connection refused to {url}")
        return {"status": "error", "error": "Service unavailable"}
    except Exception as e:
        logger.error(f"[SERVICE] Error calling {url}: {e}")
        return {"status": "error", "error": str(e)}


async def call_service_safe(
    service_url: str,
    endpoint: str,
    method: str = "POST",
    data: dict = None,
    timeout: float = 60.0,
    service_name: str = "unknown"
) -> Tuple[str, dict]:
    """Call service with safety wrapper. Returns (service_name, result)."""
    try:
        result = await call_service(service_url, endpoint, method, data, timeout)
        return (service_name, result)
    except asyncio.CancelledError:
        logger.info(f"[SERVICE] {service_name} cancelled (early exit)")
        return (service_name, {"status": "cancelled", "error": "Early exit - cancelled"})
    except Exception as e:
        logger.error(f"[SERVICE] {service_name} error: {e}")
        return (service_name, {"status": "error", "error": str(e)})


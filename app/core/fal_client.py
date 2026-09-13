import httpx
from app.core.config import settings


def submit_to_fal(model_id: str, input_params: dict, webhook_url: str) -> str:
    response = httpx.post(
        f"https://queue.fal.run/{model_id}",
        params={"fal_webhook": webhook_url},
        headers={"Authorization": f"Key {settings.fal_key}"},
        json=input_params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["request_id"]
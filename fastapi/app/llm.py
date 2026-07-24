import httpx
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _parse_model_list(env_value: str | None) -> list[str]:
    return [m.strip() for m in (env_value or "").split(",") if m.strip()]


# OPENROUTER_MODELS is a comma-separated priority list; the first model
# that OpenRouter accepts is used. Falls back to the older single-model
# OPENROUTER_MODEL var for backward compatibility.
OPENROUTER_MODELS = _parse_model_list(
    os.getenv("OPENROUTER_MODELS") or os.getenv("OPENROUTER_MODEL")
)

# Separate, cheaper/faster model list for lightweight classification
# calls (e.g. escalation-intent detection) that don't need the quality
# of the main answer-generation model. Falls back to OPENROUTER_MODELS
# if not set, so classification still works out of the box.
OPENROUTER_CLASSIFIER_MODELS = _parse_model_list(
    os.getenv("OPENROUTER_CLASSIFIER_MODELS")
) or OPENROUTER_MODELS

# Status codes where trying the next model in the list is worthwhile:
# 404 (model retired/renamed), 429 (rate-limited), 503 (upstream unavailable).
_RETRYABLE_STATUS_CODES = {404, 429, 503}


DEFAULT_SYSTEM_PROMPT = "You are a helpful agricultural assistant."


async def generate_response(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    models: list[str] | None = None,
):
    models = models if models is not None else OPENROUTER_MODELS

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    if not models:
        raise RuntimeError("OPENROUTER_MODELS (or OPENROUTER_MODEL) is not set.")

    last_error = None

    async with httpx.AsyncClient() as client:
        for model in models:
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )

            data = res.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            print(f"OpenRouter error response for model '{model}' (status {res.status_code}): {data}")
            last_error = RuntimeError(
                f"OpenRouter request failed for model '{model}' (status {res.status_code}): "
                f"{data.get('error', data)}"
            )

            if res.status_code not in _RETRYABLE_STATUS_CODES:
                raise last_error

    raise last_error
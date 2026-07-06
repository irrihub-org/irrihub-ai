import httpx
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")


async def generate_response(prompt: str):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    if not OPENROUTER_MODEL:
        raise RuntimeError("OPENROUTER_MODEL is not set.")

    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful agricultural assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

    data = res.json()

    if "choices" not in data:
        
        print(f"OpenRouter error response (status {res.status_code}): {data}")
        raise RuntimeError(
            f"OpenRouter request failed (status {res.status_code}): "
            f"{data.get('error', data)}"
        )

    return data["choices"][0]["message"]["content"]
# IrriHub AI Chatbot

FastAPI intelligence layer and KB admin portal for IrriHub.

## Services

| Service | Directory | Port |
|---|---|---|
| FastAPI Intelligence Layer | `fastapi/` | 8000 |
| KB Admin Portal | `kb-portal/` | 8001 |

## Development

See the Developer Handoff Guide for full documentation.

```bash
git clone https://github.com/irrihub-org/irrihub-ai.git
cd irrihub-ai
cd fastapi && cp .env.example .env && docker compose up --build
```

## Workflow

1. Branch: `git checkout -b feature/your-feature`
2. Develop and test locally
3. Push and open a PR
4. Lead Developer reviews, merges, and deploys

from fastapi import FastAPI
app = FastAPI(title="IrriHub AI", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "irrihub-fastapi", "version": "0.1.0"}

@app.get("/")
async def root():
    return {"message": "IrriHub AI — endpoint not configured yet"}

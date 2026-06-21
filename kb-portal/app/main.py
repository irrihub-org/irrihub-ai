from fastapi import FastAPI
app = FastAPI(title="IrriHub KB Portal", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "irrihub-kb-portal", "version": "0.1.0"}

@app.get("/")
async def root():
    return {"message": "IrriHub KB Admin Portal — not configured yet"}

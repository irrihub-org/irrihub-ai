import os, uuid, csv, io
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL")
UPLOAD_DIR = "/tmp/kb-uploads"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
db_pool = None
embed_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, embed_model
    print("Starting KB Portal...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    print("  DB connected")
    print("  Loading embedding model (~30s first run)...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded: {MODEL_NAME}")
    yield
    if db_pool: await db_pool.close()

app = FastAPI(title="IrriHub KB Admin", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def extract_text(filepath, filename):
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f: return f.read()
    elif ext == ".csv":
        rows = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                rows.append(" | ".join(f"{k}: {v}" for k,v in row.items() if v and v.strip()))
        return "\n\n".join(rows)
    elif ext == ".pdf":
        import fitz
        doc = fitz.open(filepath)
        t = "\n".join(p.get_text() for p in doc)
        doc.close()
        return t
    elif ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(filepath).paragraphs if p.text.strip())
    raise ValueError(f"Unsupported: {ext}")

def chunk_text(text):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        c = " ".join(words[start:start+CHUNK_SIZE])
        if c.strip() and len(c.strip()) > 20: chunks.append(c.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_chunks(chunks):
    return embed_model.encode(chunks).tolist()

@app.get("/health")
async def health():
    try:
        async with db_pool.acquire() as conn:
            r = await conn.fetchrow("SELECT COUNT(*) AS c FROM kb_documents")
            return {"status":"ok","service":"kb-portal","db":"connected","chunks":r["c"]}
    except: return {"status":"degraded","db":"disconnected"}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    sources, stats = [], {"total_sources":0,"total_chunks":0,"active_sources":0}
    try:
        async with db_pool.acquire() as conn:
            sources = [dict(r) for r in await conn.fetch("""
                SELECT ks.id,ks.filename,ks.category,ks.language,ks.version,ks.is_active,ks.uploaded_at,
                COUNT(kd.id) AS chunk_count FROM kb_sources ks
                LEFT JOIN kb_documents kd ON kd.source_id=ks.id GROUP BY ks.id ORDER BY ks.uploaded_at DESC""")]
            stats = dict(await conn.fetchrow("""SELECT
                (SELECT COUNT(*) FROM kb_sources) AS total_sources,
                (SELECT COUNT(*) FROM kb_documents) AS total_chunks,
                (SELECT COUNT(*) FROM kb_sources WHERE is_active=true) AS active_sources"""))
    except Exception as e: print(f"Error: {e}")
    return templates.TemplateResponse("dashboard.html",{"request":request,"sources":sources,"stats":stats})

@app.post("/upload")
async def upload_document(file: UploadFile=File(...), category: str=Form(...), language: str=Form("en")):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf",".docx",".csv",".txt"}: raise HTTPException(400,f"Unsupported: {ext}")
    contents = await file.read()
    if len(contents) > 50*1024*1024: raise HTTPException(400,"Max 50MB")
    tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(tmp,"wb") as f: f.write(contents)
    try:
        raw = extract_text(tmp, file.filename)
        if not raw.strip(): raise HTTPException(400,"No text extracted")
        chunks = chunk_text(raw)
        if not chunks: raise HTTPException(400,"No usable chunks")
        embeddings = embed_chunks(chunks)
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT id,version FROM kb_sources WHERE filename=$1 AND is_active=true", file.filename)
            ver = 1
            if existing:
                await conn.execute("UPDATE kb_sources SET is_active=false WHERE id=$1", existing["id"])
                await conn.execute("DELETE FROM kb_documents WHERE source_id=$1", existing["id"])
                ver = existing["version"]+1
            sid = str(uuid.uuid4())
            await conn.execute("INSERT INTO kb_sources(id,filename,version,category,language,uploader_id,uploaded_at) VALUES($1,$2,$3,$4,$5,$6,$7)",
                sid, file.filename, ver, category, language, "admin-portal", datetime.utcnow())
            for ct,emb in zip(chunks, embeddings):
                es = "["+",".join(str(x) for x in emb)+"]"
                await conn.execute("INSERT INTO kb_documents(source_id,chunk_text,embedding,category,language) VALUES($1,$2,$3::vector,$4,$5)",
                    sid, ct, es, category, language)
        return RedirectResponse(url=f"/?success={file.filename}&chunks={len(chunks)}&version={ver}", status_code=303)
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.post("/delete/{source_id}")
async def delete_source(source_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE kb_sources SET is_active=false WHERE id=$1", source_id)
    return RedirectResponse(url="/?deleted=true", status_code=303)

@app.post("/reactivate/{source_id}")
async def reactivate_source(source_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE kb_sources SET is_active=true WHERE id=$1", source_id)
    return RedirectResponse(url="/?reactivated=true", status_code=303)

@app.get("/search")
async def search_kb(q: str=""):
    if not q or len(q)<2: return JSONResponse({"results":[]})
    qe = embed_model.encode([q])[0].tolist()
    es = "["+",".join(str(x) for x in qe)+"]"
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""SELECT kd.chunk_text,kd.category,kd.language,ks.filename,ks.version,
            kd.embedding<=>$1::vector AS distance FROM kb_documents kd
            JOIN kb_sources ks ON kd.source_id=ks.id WHERE ks.is_active=true
            ORDER BY distance ASC LIMIT 5""", es)
    return JSONResponse({"results":[{"chunk":r["chunk_text"][:300],"source":r["filename"],
        "category":r["category"],"language":r["language"],"distance":round(r["distance"],4),
        "version":r["version"]} for r in rows]})

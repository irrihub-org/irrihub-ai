# IrriHub Knowledge Base Content

Drop documents here for ingestion into the KB.

## Folders
- `manuals/` — Hardware installation and maintenance guides
- `faqs/` — Frequently asked questions from farmers
- `product-sheets/` — Product specifications and pricing

## Supported Formats
PDF, DOCX, CSV, TXT

## How content gets into the KB
1. Documents are placed here in the repo
2. The KB Admin Portal (admin.irrihub.com) ingests them
3. They are chunked, embedded, and stored in pgvector
4. FastAPI queries them during RAG retrieval

## For Development
Copy these files to your local environment to test
the ingestion and RAG pipeline locally.

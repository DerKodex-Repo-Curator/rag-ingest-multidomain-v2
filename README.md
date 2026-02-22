# RAG Ingest Multi-Domain

Multi-domain RAG (Retrieval Augmented Generation) ingestion pipeline that syncs documents from Microsoft SharePoint/OneDrive, generates embeddings, and stores them in FAISS indexes.

## Overview

This project is a document ingestion pipeline that:

1. **Connects to Microsoft SharePoint/OneDrive** via Microsoft Graph API using OAuth2 client credentials
2. **Monitors multiple domain folders** (configurable via `DOMAINS` environment variable)
3. **Downloads documents** (PDF, DOCX, TXT) from SharePoint
4. **Extracts and cleans text** from those documents
5. **Chunks text** using a parent-child splitting strategy:
   - Parent chunks: 1200 chars with 100 char overlap
   - Child chunks: 300 chars with 50 char overlap
6. **Generates embeddings** using the `nomic-ai/nomic-embed-text-v1.5` model
7. **Stores vectors in FAISS indexes** - one index per domain
8. **Runs continuously**, polling for file changes at regular intervals (default 300s)

## Key Features

- **Change detection**: Detects new, modified, and deleted files and rebuilds indexes accordingly
- **Multi-tenant**: Uses `TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET` for Microsoft Graph auth
- **CPU-optimized**: Runs on CPU (can be reverted to GPU)
- **Per-domain isolation**: Each domain has its own FAISS index, metadata, and chunks files

## Requirements

- Python 3.10
- Microsoft SharePoint/OneDrive with app registration

## Configuration

Environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `TENANT_ID` | Microsoft Azure AD tenant ID |
| `CLIENT_ID` | Application (client) ID |
| `CLIENT_SECRET` | Client secret for authentication |
| `SITE_ID` | SharePoint site ID |
| `DRIVE_ID` | SharePoint drive ID |
| `DOMAINS` | Comma-separated list of folder names to monitor (e.g., `Councilreports,Humanresources`) |
| `POLL_INTERVAL` | How often to check for changes in seconds (default: 300) |

## Quick Start

1. Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

2. Build and run with Docker:

```bash
docker-compose up -d
```

## Project Structure

```
.
├── app/
│   └── ingest.py          # Main ingestion script
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables
└── README.md              # This file
```

## Output

For each domain, the following files are created:

- `{domain}/faiss_index.bin` - FAISS vector index
- `{domain}/metadata.json` - File metadata (names, modification dates)
- `{domain}/chunks.json` - Text chunks with source information

## Docker

### Build

```bash
docker build -t rag-ingest-multidomain .
```

### Run

```bash
docker run -d --env-file .env rag-ingest-multidomain
```

## GPU Support

The project is configured for CPU by default. To enable GPU:

1. Uncomment `torch` in `requirements.txt`
2. In `Dockerfile`, comment out the CPU-only PyTorch install line
3. In `app/ingest.py`, uncomment the CUDA device detection and model movement lines

## License

MIT

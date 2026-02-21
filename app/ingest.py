import os
import time
import json
import requests
import faiss
import logging
import traceback
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from pypdf import PdfReader
import docx

# Phase 1: Better Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ingestion")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SITE_ID = os.getenv("SITE_ID")
DRIVE_ID = os.getenv("DRIVE_ID")

# Split comma-separated string into a list. Strip whitespace.
raw_domains = os.getenv("DOMAINS", "Councilreports")
DOMAINS = [d.strip() for d in raw_domains.split(",") if d.strip()]

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 300))

# ---------------------------------------------------------
# PATH HELPERS (MULTI-DOMAIN)
# ---------------------------------------------------------
def get_domain_paths(domain_name):
    """
    Creates a subdirectory for the domain if it doesn't exist
    and returns the paths for index, metadata, and chunks.
    """
    # Create folder: e.g., /app/Humanresources
    base_dir = Path(domain_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "index": str(base_dir / "faiss_index.bin"),
        "metadata": str(base_dir / "metadata.json"),
        "chunks": str(base_dir / "chunks.json")
    }

# ---------------------------------------------------------
# AUTHENTICATION & GRAPH API
# ---------------------------------------------------------
def get_access_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def list_files(token, folder_path):
    # Dynamic folder path in URL
    url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/root:/{folder_path}:/children"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["value"]

def download_file(token, item_id):
    url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives/{DRIVE_ID}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.content

# ---------------------------------------------------------
# TEXT EXTRACTION & CLEANING
# ---------------------------------------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.replace('\n \n', '\n')
    text = text.replace('\r', '\n')
    text = re.sub(' +', ' ', text.strip())
    text = re.sub('\n+', '\n', text)
    return text

def extract_text_from_pdf(data):
    temp = f"/tmp/temp_{os.getpid()}.pdf"
    try:
        with open(temp, "wb") as f:
            f.write(data)
        reader = PdfReader(temp)
        pages_text = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            cleaned = clean_text(text)
            if cleaned:
                pages_text.append((page_num, cleaned))
        return pages_text
    finally:
        if os.path.exists(temp):
            os.remove(temp)

def extract_text_from_docx(data):
    temp = f"/tmp/temp_{os.getpid()}.docx"
    try:
        with open(temp, "wb") as f:
            f.write(data)
        doc = docx.Document(temp)
        text = "\n".join([p.text for p in doc.paragraphs])
        return [(None, clean_text(text))]
    finally:
        if os.path.exists(temp):
            os.remove(temp)

def extract_text(filename, data):
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(data)
    if filename.lower().endswith(".docx"):
        return extract_text_from_docx(data)
    if filename.lower().endswith(".txt"):
        return [(None, clean_text(data.decode("utf-8", errors="ignore")))]
    return []

# ---------------------------------------------------------
# EMBEDDINGS (CPU OPTIMIZED)
# ---------------------------------------------------------
# 1. Detect Device
# GPU REVERT: Uncomment the line below and comment out 'device = "cpu"'
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
log.info(f"Using device: {device.upper()}")

log.info("Loading embedding model (nomic-embed-text-v1.5)...")
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# 2. Move Model
# GPU REVERT: Uncomment the line below
# model.to(device)
model.eval()

# Get embedding dimension
with torch.no_grad():
    # Create dummy input
    test_input = tokenizer("test", return_tensors="pt", truncation=True, padding=True)
    # GPU REVERT: Uncomment the line below
    # test_input = {k: v.to(device) for k, v in test_input.items()}
    
    test_output = model(**test_input)
    EMBEDDING_DIM = test_output.last_hidden_state.shape[-1]

log.info(f"Embedding model loaded on {device}. Dimension: {EMBEDDING_DIM}")

def embed(text):
    if not text or not text.strip():
        return torch.zeros(EMBEDDING_DIM).numpy()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # GPU REVERT: Uncomment the line below
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embedding = (sum_embeddings / sum_mask).squeeze()
    return embedding.cpu().numpy()

# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

# ---------------------------------------------------------
# FAISS & STORAGE (PATH AWARE)
# ---------------------------------------------------------
def load_index(path):
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatL2(EMBEDDING_DIM)

def save_index(index, path):
    faiss.write_index(index, path)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------------------------------------
# INDEX REBUILD
# ---------------------------------------------------------
def rebuild_index_for_domain(token, files, domain_name, paths):
    log.warning(f"[{domain_name}] Rebuilding FAISS index from scratch...")
    
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = {}
    chunks = {}
    
    vector_id = 0
    
    for f in files:
        filename = f["name"]
        modified = f["lastModifiedDateTime"]
        item_id = f["id"]
        
        log.info(f"[{domain_name}] Processing: {filename}")
        
        try:
            data = download_file(token, item_id)
            pages_text = extract_text(filename, data)
            
            chunk_count = 0
            
            for page_num, page_text in pages_text:
                parent_chunks = parent_splitter.split_text(page_text)
                
                for parent_text in parent_chunks:
                    page_str = f"Page {page_num}" if page_num else "Whole Document"
                    full_parent_context = f"Document: {filename}\n{page_str}\n\n{parent_text}"
                    
                    child_chunks = child_splitter.split_text(parent_text)
                    
                    for child_text in child_chunks:
                        # Include domain in the embedding context if desired, 
                        # though physical separation usually suffices.
                        full_child_search_text = f"Document: {filename}\nContent: {child_text}"
                        
                        vec = embed(full_child_search_text)
                        index.add(vec.reshape(1, -1))
                        
                        chunks[str(vector_id)] = {
                            "text": full_parent_context,
                            "child_text": child_text,
                            "source": filename,
                            "page": str(page_num) if page_num else "",
                            "domain": domain_name
                        }
                        
                        vector_id += 1
                        chunk_count += 1
            
            log.info(f"[{domain_name}] {filename}: {chunk_count} vectors created.")
            metadata[filename] = modified
            
        except Exception as e:
            log.error(f"[{domain_name}] Error processing {filename}: {e}")
            log.error(traceback.format_exc())
            continue
    
    save_index(index, paths["index"])
    save_json(metadata, paths["metadata"])
    save_json(chunks, paths["chunks"])
    
    log.info(f"[{domain_name}] Rebuild complete. Total vectors: {vector_id}")

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def run_ingestion():
    log.info(f"Monitoring Domains: {DOMAINS}")
    
    while True:
        try:
            log.info("Starting poll cycle...")
            token = get_access_token()
            
            # Iterate over each configured domain
            for domain in DOMAINS:
                try:
                    paths = get_domain_paths(domain)
                    
                    # Load current state for this domain
                    metadata = load_json(paths["metadata"])
                    
                    log.info(f"[{domain}] Checking SharePoint folder...")
                    files = list_files(token, domain)
                    
                    current_files = {f["name"]: f for f in files}
                    previous_files = set(metadata.keys())
                    current_file_names = set(current_files.keys())
                    
                    # Detect Deletions
                    deleted_files = previous_files - current_file_names
                    if deleted_files:
                        log.warning(f"[{domain}] Deleted files detected: {deleted_files}")
                        rebuild_index_for_domain(token, files, domain, paths)
                        continue
                    
                    # Detect Modifications / Additions
                    files_changed = False
                    for filename, f in current_files.items():
                        modified = f["lastModifiedDateTime"]
                        
                        if filename not in metadata:
                            log.info(f"[{domain}] New file detected: {filename}")
                            files_changed = True
                            break
                        elif metadata[filename] < modified:
                            log.warning(f"[{domain}] File modified: {filename}")
                            files_changed = True
                            break
                    
                    if files_changed:
                        rebuild_index_for_domain(token, files, domain, paths)
                    else:
                        log.info(f"[{domain}] No changes.")
                        
                except Exception as e:
                    log.error(f"[{domain}] Failed to process domain.")
                    log.error(traceback.format_exc())
        
        except Exception as e:
            log.error("Error during global polling cycle:")
            log.error(traceback.format_exc())
        
        log.info(f"Sleeping {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run_ingestion()
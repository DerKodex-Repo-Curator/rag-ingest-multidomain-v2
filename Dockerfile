# Changed to slim to significantly reduce image size
FROM python:3.10-slim
# FROM python:3.10  <-- Uncomment this if you revert to GPU to ensure full driver support

WORKDIR /app

# Install system dependencies required for building some python packages (needed for slim)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only PyTorch first (Significant size reduction)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
# We remove --no-cache-dir to speed up rebuilds if you edit code often, 
# but keep it if you want to save space.
RUN pip install --no-cache-dir -r requirements.txt

# --- GPU REVERSION INSTRUCTIONS ---
# To revert to GPU:
# 1. Uncomment 'torch' in requirements.txt
# 2. Comment out the "Install CPU-only PyTorch" line above
# 3. Run: RUN pip install --no-cache-dir -r requirements.txt
# ----------------------------------

COPY . .

CMD ["python", "ingest.py"]

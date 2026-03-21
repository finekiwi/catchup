FROM python:3.12-slim

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# System dependencies for docling + chromadb + PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    git \
    build-essential \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (prevents docling from pulling 2.5GB GPU build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies via pyproject.toml with uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy app code
COPY . .

# Create writable data directories
RUN mkdir -p data/chroma data/logs data/parsed data/figures data/image_uploads

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/demo.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]

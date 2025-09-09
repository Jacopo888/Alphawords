# AlphaScrabble Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    qtbase5-dev \
    libqt5core5a \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install AlphaScrabble
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p data/selfplay data/training checkpoints logs lexica_cache

# Download and compile lexicon (optional, can be done at runtime)
RUN if [ -f "scripts/setup_lexicon.sh" ]; then \
    chmod +x scripts/setup_lexicon.sh && \
    ./scripts/setup_lexicon.sh; \
    fi

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Set default command
CMD ["alphascrabble", "--help"]

# Labels
LABEL maintainer="AlphaScrabble Team"
LABEL description="AlphaZero-style Scrabble engine with MCTS and neural networks"
LABEL version="0.1.0"

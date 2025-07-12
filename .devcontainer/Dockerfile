# Base development image - Go will be installed via devcontainer.json features
FROM mcr.microsoft.com/vscode/devcontainers/base:bullseye AS base

# Install system dependencies for neural network development
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        pkg-config \
        libopenblas-dev \
        liblapack-dev \
        libomp-dev \
        libfftw3-dev \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        nodejs \
        npm \
        curl \
        wget \
        git \
        htop \
        vim \
        jq \
        tree \
        unzip \
        && apt-get autoremove -y && apt-get clean -y

# Note: Go development tools will be installed via postCreateCommand after Go feature is installed

# Development stage
FROM base AS development

# Install Python dependencies for ML comparison and benchmarking
# Note: Minimal setup for ARM64 Linux containers (TensorFlow not available)
RUN python3 -m pip install --upgrade pip \
    && pip3 install --no-cache-dir \
        numpy \
        matplotlib \
        jupyter \
        memory-profiler \
        psutil

# Install Node.js tools for dashboard and visualization
RUN npm install -g \
    http-server

# Setup cache directories
RUN mkdir -p /home/vscode/.cache/go-build \
    && mkdir -p /home/vscode/go/pkg/mod \
    && chown -R vscode:vscode /home/vscode/.cache \
    && chown -R vscode:vscode /home/vscode/go

# Switch to vscode user
USER vscode

# Setup bash aliases for convenience
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias cc="claude-code"' >> ~/.bashrc \
    && echo 'alias bee="go run ./cmd/bee"' >> ~/.bashrc \
    && echo 'alias make-help="make help"' >> ~/.bashrc

# GPU development stage (extends development)
FROM development AS gpu-development

# Note: ARM64 Linux containers do not support NVIDIA CUDA
# Using CPU-only libraries for cross-platform compatibility
USER vscode

# Install PyTorch CPU-only version for ARM64 compatibility
RUN pip3 install --upgrade --no-cache-dir \
    torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Production stage (minimal runtime)
FROM golang:1.24-bullseye AS production

WORKDIR /app

# Copy Go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN make build

# Expose default port
EXPOSE 8080

# Run the application
CMD ["./bin/bee", "serve"]
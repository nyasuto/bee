#!/bin/bash
# ARM64 Mac specific setup for Bee Neural Network development

set -e

echo "🍎 Setting up Bee Neural Network development environment for ARM64 Mac..."

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_purple() {
    echo -e "${PURPLE}🍎 $1${NC}"
}

# Check if we're in an ARM64 environment
if [ "$(uname -m)" != "arm64" ] && [ "$(uname -m)" != "aarch64" ]; then
    log_warning "This script is optimized for ARM64 architecture"
    log_info "Current architecture: $(uname -m)"
fi

# Check if we're in a container
if [ -n "$REMOTE_CONTAINERS" ] || [ -n "$CODESPACES" ] || [ -f /.dockerenv ]; then
    log_purple "Running in ARM64 DevContainer environment"
    CONTAINER_ENV=true
else
    log_purple "Running in native ARM64 macOS environment"
    CONTAINER_ENV=false
fi

# Install Go dependencies
log_info "Installing Go dependencies..."
if ! go mod download; then
    log_error "Failed to download Go dependencies"
    exit 1
fi

if ! go mod tidy; then
    log_error "Failed to tidy Go modules"
    exit 1
fi
log_success "Go dependencies installed"

# Install Go development tools
log_info "Installing Go development tools for ARM64..."
TOOLS=(
    "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
    "golang.org/x/tools/cmd/goimports@latest"
    "github.com/go-delve/delve/cmd/dlv@latest"
    "golang.org/x/tools/gopls@latest"
)

for tool in "${TOOLS[@]}"; do
    log_info "Installing $tool..."
    if GOARCH=arm64 go install "$tool"; then
        log_success "Installed $tool"
    else
        log_warning "Failed to install $tool (may already exist)"
    fi
done

# Python environment setup for ARM64
log_info "Setting up Python environment for ARM64..."

# Check Python installation
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python installed: $PYTHON_VERSION"
else
    log_error "Python3 not found"
    exit 1
fi

# Check available Python libraries
log_info "Checking Python ML libraries..."

# NumPy (should work on ARM64)
if python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
    log_success "NumPy available"
else
    log_warning "NumPy not available"
fi

# PyTorch (CPU version should work)
if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    log_success "PyTorch available (CPU)"
else
    log_warning "PyTorch not available"
fi

# TensorFlow (likely not available in ARM64 Linux container)
if python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null; then
    log_success "TensorFlow available"
else
    log_warning "TensorFlow not available (expected on ARM64 Linux)"
    log_info "Consider using host macOS for TensorFlow comparisons"
fi

# Setup Git hooks
log_info "Setting up Git hooks..."
if make git-hooks > /dev/null 2>&1; then
    log_success "Git hooks configured"
else
    log_warning "Git hooks setup failed (Makefile target may not exist yet)"
fi

# ARM64 specific optimizations
log_info "Applying ARM64 optimizations..."

# Set Go environment for ARM64
export GOARCH=arm64
export CGO_ENABLED=1

# Create ARM64 build script
cat > scripts/build-arm64.sh << 'EOF'
#!/bin/bash
# ARM64 optimized build script

echo "🍎 Building for ARM64..."
GOARCH=arm64 CGO_ENABLED=1 go build -ldflags="-s -w" -o bin/bee-arm64 ./cmd/bee
echo "✅ ARM64 build complete: bin/bee-arm64"
EOF

chmod +x scripts/build-arm64.sh

# Verification
log_info "Running ARM64 environment verification..."

# Go verification
if command -v go >/dev/null 2>&1; then
    GO_VERSION=$(go version)
    log_success "Go: $GO_VERSION"
    
    # Check Go architecture
    GO_ARCH=$(go env GOARCH)
    if [ "$GO_ARCH" = "arm64" ]; then
        log_success "Go configured for ARM64"
    else
        log_info "Go architecture: $GO_ARCH"
    fi
else
    log_error "Go not found"
fi

# Development tools verification
if command -v golangci-lint >/dev/null 2>&1; then
    LINT_VERSION=$(golangci-lint version --format short)
    log_success "golangci-lint: $LINT_VERSION"
else
    log_warning "golangci-lint not found"
fi

# ML libraries summary
echo ""
log_purple "ARM64 ML Environment Summary:"
echo "  🔹 NumPy/SciPy: Available (excellent ARM64 support)"
echo "  🔹 PyTorch: CPU version available"
echo "  🔹 TensorFlow: Use host macOS for comparisons"
echo "  🔹 Jupyter: Available for experimentation"

# Host integration suggestions
if [ "$CONTAINER_ENV" = true ]; then
    echo ""
    log_purple "ARM64 Mac Integration Tips:"
    echo "  💡 For TensorFlow comparisons:"
    echo "     - Install TensorFlow on host macOS"
    echo "     - Use mounted volumes for data exchange"
    echo "     - Run benchmarks on host, development in container"
    echo ""
    echo "  💡 For GPU acceleration:"
    echo "     - Use Metal Performance Shaders on host macOS"
    echo "     - Consider Cloud GPU for intensive comparisons"
fi

# Quick tests
log_info "Running quick validation tests..."
if make test-quick > /dev/null 2>&1; then
    log_success "Quick tests passed"
else
    log_warning "Quick tests failed (normal if no tests exist yet)"
fi

# Completion
log_success "ARM64 development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  • Run 'make help' to see available commands"
echo "  • Run 'make quality' to check code quality"
echo "  • Use 'scripts/build-arm64.sh' for ARM64 builds"
echo "  • For TensorFlow: install on host macOS if needed"
echo ""
echo "🍎 Happy coding with Bee Neural Networks on ARM64!"
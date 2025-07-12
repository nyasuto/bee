#!/bin/bash
# Bee development environment setup script

set -e

echo "🐝 Setting up Bee Neural Network development environment..."

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if we're in DevContainer
if [ -n "$REMOTE_CONTAINERS" ] || [ -n "$CODESPACES" ]; then
    log_info "Running in containerized environment"
    CONTAINER_ENV=true
else
    log_info "Running in local environment"
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

# Install development tools
log_info "Installing Go development tools..."
TOOLS=(
    "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
    "golang.org/x/tools/cmd/goimports@latest"
    "github.com/go-delve/delve/cmd/dlv@latest"
    "golang.org/x/tools/gopls@latest"
)

for tool in "${TOOLS[@]}"; do
    log_info "Installing $tool..."
    if go install "$tool"; then
        log_success "Installed $tool"
    else
        log_warning "Failed to install $tool (may already exist)"
    fi
done

# Setup Git hooks
log_info "Setting up Git hooks..."
if make git-hooks > /dev/null 2>&1; then
    log_success "Git hooks configured"
else
    log_warning "Git hooks setup failed (Makefile target may not exist yet)"
fi

# Verify installation
log_info "Verifying installation..."
if command -v go >/dev/null 2>&1; then
    GO_VERSION=$(go version | cut -d' ' -f3)
    log_success "Go installed: $GO_VERSION"
else
    log_error "Go not found"
fi

if command -v golangci-lint >/dev/null 2>&1; then
    LINT_VERSION=$(golangci-lint version --format short)
    log_success "golangci-lint installed: $LINT_VERSION"
else
    log_warning "golangci-lint not found"
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python installed: $PYTHON_VERSION"
else
    log_warning "Python3 not found"
fi

# Check Python ML libraries
if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    log_success "PyTorch available"
else
    log_warning "PyTorch not available"
fi

if python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null; then
    log_success "TensorFlow available"
else
    log_warning "TensorFlow not available"
fi

# Run quick tests
log_info "Running quick validation tests..."
if make test-quick > /dev/null 2>&1; then
    log_success "Quick tests passed"
else
    log_warning "Quick tests failed (normal if no tests exist yet)"
fi

# Setup completion
log_success "Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  • Run 'make help' to see available commands"
echo "  • Run 'make quality' to check code quality"
echo "  • Run 'claude-code' to start AI-assisted development"
echo ""
echo "🐝 Happy coding with Bee Neural Networks!"
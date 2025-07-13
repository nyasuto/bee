# Bee Neural Network Project Makefile
# 段階的ニューラルネットワーク学習プロジェクト開発自動化

.PHONY: help install dev build clean test lint format quality quality-fix validate analyze setup git-hooks env-info pr-ready install-tools verify-env setup-native

# Default target
.DEFAULT_GOAL := help

# Help target
help: ## Show this help message
	@echo "$(CYAN)🐝 Bee Neural Network Project - Development Commands$(NC)"
	@echo "$(BLUE)==============================================$(NC)"
	@echo ""
	@echo "$(YELLOW)🚀 Quick Start:$(NC)"
	@echo "  make setup    - プロジェクト初期セットアップ"
	@echo "  make dev      - 開発環境開始"
	@echo "  make quality  - 品質チェック実行"
	@echo ""
	@echo "$(YELLOW)📋 Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

# Development Setup
setup: ## プロジェクト初期セットアップ (依存関係インストール + git hooks)
	@echo "$(CYAN)🐝 Bee プロジェクトをセットアップ中...$(NC)"
	@$(MAKE) install
	@$(MAKE) git-hooks
	@echo "$(GREEN)✅ セットアップ完了! 'make setup-native' で開発ツールもインストールできます。$(NC)"

install: ## Go依存関係をインストール
	@echo "$(BLUE)📦 Go 依存関係をインストール中...$(NC)"
	@if [ -f go.mod ]; then \
		go mod tidy; \
		go mod download; \
	else \
		echo "$(YELLOW)⚠️  go.mod が見つかりません。プロジェクト初期化が必要です。$(NC)"; \
		echo "$(BLUE)Go module初期化中...$(NC)"; \
		go mod init github.com/user/bee; \
	fi
	@echo "$(GREEN)✅ 依存関係のインストール完了$(NC)"

# Development Commands
dev: ## 開発環境開始
	@echo "$(CYAN)🚀 開発環境を開始中...$(NC)"
	@echo "$(BLUE)開発用ツールを準備中...$(NC)"
	@if [ -f cmd/bee/main.go ]; then \
		echo "$(GREEN)bee CLI ツールをビルド中...$(NC)"; \
		go build -o bin/bee ./cmd/bee; \
		echo "$(GREEN)✅ bee CLI ready: ./bin/bee$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  cmd/bee/main.go が見つかりません。まずPhase 1を実装してください。$(NC)"; \
	fi

build: ## プロダクション用ビルド
	@echo "$(BLUE)🏗️  プロダクション用ビルド中...$(NC)"
	@mkdir -p bin
	@if [ -f cmd/bee/main.go ]; then \
		go build -ldflags="-s -w" -o bin/bee ./cmd/bee; \
		echo "$(GREEN)✅ ビルド完了: bin/bee$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  cmd/bee/main.go が見つかりません$(NC)"; \
	fi

clean: ## ビルド成果物とキャッシュをクリーン
	@echo "$(YELLOW)🧹 ビルド成果物をクリーン中...$(NC)"
	@rm -rf bin/
	@rm -rf coverage.out
	@rm -rf coverage.html
	@go clean -cache -testcache -modcache
	@echo "$(GREEN)✅ クリーンアップ完了$(NC)"

# Testing
test: ## テスト実行
	@echo "$(BLUE)🧪 テスト実行中...$(NC)"
	@if ls *_test.go >/dev/null 2>&1 || find . -name "*_test.go" -type f | grep -q .; then \
		go test -v ./...; \
	else \
		echo "$(YELLOW)⚠️  テストファイルが見つかりません$(NC)"; \
	fi

test-coverage: ## カバレッジ付きテスト実行
	@echo "$(BLUE)🧪 カバレッジ付きテスト実行中...$(NC)"
	@if ls *_test.go >/dev/null 2>&1 || find . -name "*_test.go" -type f | grep -q .; then \
		go test -coverprofile=coverage.out ./...; \
		go tool cover -html=coverage.out -o coverage.html; \
		echo "$(GREEN)📊 カバレッジレポート: coverage.html$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  テストファイルが見つかりません$(NC)"; \
	fi

benchmark: ## ベンチマークテスト実行
	@echo "$(BLUE)⚡ ベンチマークテスト実行中...$(NC)"
	@if find . -name "*_test.go" -exec grep -l "Benchmark" {} \; | grep -q .; then \
		go test -bench=. -benchmem ./...; \
	else \
		echo "$(YELLOW)⚠️  ベンチマークテストが見つかりません$(NC)"; \
	fi

# Code Quality
lint: ## リント実行
	@echo "$(BLUE)🔍 リント実行中...$(NC)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	elif command -v golint >/dev/null 2>&1; then \
		golint ./...; \
	else \
		echo "$(YELLOW)⚠️  golangci-lint または golint が見つかりません$(NC)"; \
		echo "$(BLUE)go vet で基本チェックを実行中...$(NC)"; \
		go vet ./...; \
	fi

format: ## コードフォーマット
	@echo "$(BLUE)💅 コードフォーマット中...$(NC)"
	@go fmt ./...
	@if command -v goimports >/dev/null 2>&1; then \
		goimports -w .; \
	fi
	@echo "$(GREEN)✅ フォーマット完了$(NC)"

format-check: ## フォーマットチェック
	@echo "$(BLUE)💅 フォーマットチェック中...$(NC)"
	@if [ -n "$$(gofmt -l .)" ]; then \
		echo "$(RED)❌ フォーマットされていないファイル:$(NC)"; \
		gofmt -l .; \
		exit 1; \
	else \
		echo "$(GREEN)✅ フォーマット確認完了$(NC)"; \
	fi

# Comprehensive Quality Checks
quality: ## 全品質チェック実行 (lint + format + test)
	@echo "$(CYAN)🔍 包括的品質チェック実行中...$(NC)"
	@echo "$(BLUE)1/4 フォーマットチェック...$(NC)"
	@$(MAKE) format-check
	@echo "$(BLUE)2/4 リント...$(NC)"
	@$(MAKE) lint
	@echo "$(BLUE)3/4 go vet...$(NC)"
	@go vet ./...
	@echo "$(BLUE)4/4 テスト...$(NC)"
	@$(MAKE) test
	@echo "$(GREEN)✅ 品質チェック完了$(NC)"

quality-fix: ## 品質チェック（自動修正付き）
	@echo "$(CYAN)🔧 品質チェック（自動修正）実行中...$(NC)"
	@$(MAKE) format
	@$(MAKE) quality

validate: ## プロジェクト設定と依存関係検証
	@echo "$(BLUE)✅ プロジェクト検証中...$(NC)"
	@go mod verify
	@go mod tidy
	@echo "$(GREEN)✅ 検証完了$(NC)"

# Git hooks setup
git-hooks: ## Git pre-commit hookをセットアップ
	@echo "🔗 Git pre-commit hookを設定中..."
	@mkdir -p .git/hooks
	@if [ -f .git-hooks/pre-commit ]; then \
		cp .git-hooks/pre-commit .git/hooks/pre-commit; \
		chmod +x .git/hooks/pre-commit; \
		echo "✅ Pre-commit hook設定完了 (.git-hooks/pre-commit から)"; \
	else \
    	echo '#!/bin/sh\nmake quality' > .git/hooks/pre-commit; \
		chmod +x .git/hooks/pre-commit; \
    	echo "✅ Pre-commit hook設定完了 (フォールバック)"; \
	fi

pr-ready: ## Pull Request用準備 (品質 + ビルド)
	@echo "$(CYAN)🚀 Pull Request用準備中...$(NC)"
	@$(MAKE) quality-fix
	@$(MAKE) build
	@echo "$(GREEN)✅ Pull Request準備完了!$(NC)"

# Analysis and Documentation
analyze: ## コード解析とメトリクス
	@echo "$(BLUE)📊 コード解析中...$(NC)"
	@if command -v gocyclo >/dev/null 2>&1; then \
		echo "$(BLUE)循環的複雑度チェック...$(NC)"; \
		gocyclo -over 15 .; \
	fi
	@if command -v ineffassign >/dev/null 2>&1; then \
		echo "$(BLUE)未使用変数チェック...$(NC)"; \
		ineffassign .; \
	fi
	@echo "$(BLUE)依存関係分析...$(NC)"
	@go list -m all

env-info: ## 環境情報表示
	@echo "$(CYAN)🔍 Environment Information$(NC)"
	@echo "$(BLUE)========================$(NC)"
	@echo "$(YELLOW)Go version:$(NC) $$(go version)"
	@echo "$(YELLOW)OS:$(NC) $$(uname -s)"
	@echo "$(YELLOW)Architecture:$(NC) $$(uname -m)"
	@echo "$(YELLOW)Working directory:$(NC) $$(pwd)"
	@echo "$(YELLOW)Git branch:$(NC) $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@if [ -f go.mod ]; then \
		echo "$(YELLOW)Module:$(NC) $$(head -1 go.mod | cut -d' ' -f2)"; \
	fi

# Phase-specific development
phase1: ## Phase 1 (Perceptron) 開発環境
	@echo "$(CYAN)🐝 Phase 1: Perceptron 開発環境準備中...$(NC)"
	@mkdir -p phase1 cmd/bee datasets
	@echo "$(GREEN)✅ Phase 1 ディレクトリ準備完了$(NC)"

phase2: ## Phase 2 (CNN/RNN) 開発環境
	@echo "$(CYAN)🐝 Phase 2: CNN/RNN 開発環境準備中...$(NC)"
	@mkdir -p phase2 datasets/mnist datasets/cifar
	@echo "$(GREEN)✅ Phase 2 ディレクトリ準備完了$(NC)"

phase3: ## Phase 3 (Attention/Transformer) 開発環境
	@echo "$(CYAN)🐝 Phase 3: Attention/Transformer 開発環境準備中...$(NC)"
	@mkdir -p phase3 datasets/text
	@echo "$(GREEN)✅ Phase 3 ディレクトリ準備完了$(NC)"

phase4: ## Phase 4 (LLM) 開発環境
	@echo "$(CYAN)🐝 Phase 4: LLM 開発環境準備中...$(NC)"
	@mkdir -p phase4 datasets/large
	@echo "$(GREEN)✅ Phase 4 ディレクトリ準備完了$(NC)"

# Emergency Commands
reset: ## プロジェクトをクリーン状態にリセット
	@echo "$(RED)🔄 プロジェクトをクリーン状態にリセット中...$(NC)"
	@echo "$(YELLOW)⚠️  これはgo.mod以外のすべてのビルド成果物を削除します$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) clean; \
		rm -rf vendor/; \
		echo "$(GREEN)✅ プロジェクトリセット完了$(NC)"; \
	else \
		echo "$(BLUE)ℹ️  リセットをキャンセルしました$(NC)"; \
	fi

# Status and Information
status: ## プロジェクト状態表示
	@echo "$(CYAN)📊 Bee Project Status$(NC)"
	@echo "$(BLUE)==================$(NC)"
	@echo "$(YELLOW)Git status:$(NC)"
	@git status --porcelain 2>/dev/null || echo "Not a git repository"
	@echo ""
	@echo "$(YELLOW)Go modules:$(NC)"
	@if [ -f go.mod ]; then \
		go list -m all | head -10; \
	else \
		echo "go.mod not found"; \
	fi

# Quick shortcuts for common tasks
q: quality ## Quick alias for quality checks
qf: quality-fix ## Quick alias for quality-fix
d: dev ## Quick alias for dev
b: build ## Quick alias for build
t: test ## Quick alias for test
l: lint ## Quick alias for lint
f: format ## Quick alias for format

install-tools: ## 開発ツールをインストール
	@echo "$(BLUE)🔧 Go開発ツールインストール中...$(NC)"
	@echo "golangci-lint をインストール中..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@echo "goimports をインストール中..."
	@go install golang.org/x/tools/cmd/goimports@latest
	@echo "delve debugger をインストール中..."
	@go install github.com/go-delve/delve/cmd/dlv@latest
	@echo "$(GREEN)✅ 開発ツールインストール完了$(NC)"

verify-env: ## 開発環境設定を検証
	@echo "$(BLUE)🔍 ネイティブ開発環境検証中...$(NC)"
	@echo "$(YELLOW)Go version:$(NC) $$(go version)"
	@echo "$(YELLOW)Git version:$(NC) $$(git --version)"
	@echo "$(YELLOW)Make version:$(NC) $$(make --version | head -1)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		echo "$(GREEN)✅ golangci-lint: $$(golangci-lint --version)$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  golangci-lint not installed. Run 'make install-tools'$(NC)"; \
	fi
	@if command -v goimports >/dev/null 2>&1; then \
		echo "$(GREEN)✅ goimports: available$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  goimports not installed. Run 'make install-tools'$(NC)"; \
	fi
	@if [ -f go.mod ]; then \
		go mod verify; \
		echo "$(GREEN)✅ Go modules verified$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  go.mod not found$(NC)"; \
	fi

setup-native: ## ネイティブ開発環境完全セットアップ
	@echo "$(CYAN)🐝 Bee ネイティブ開発環境セットアップ中...$(NC)"
	@$(MAKE) install
	@$(MAKE) install-tools
	@$(MAKE) git-hooks
	@$(MAKE) verify-env
	@echo "$(GREEN)✅ ネイティブ開発環境セットアップ完了!$(NC)"
	@echo "$(BLUE)📋 次のステップ:$(NC)"
	@echo "  1. 'make phase1' で Phase 1 開発環境準備"
	@echo "  2. 'make dev' で開発開始"
	@echo "  3. 'make quality' で品質チェック"
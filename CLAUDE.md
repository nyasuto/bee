# CLAUDE.md - Bee Neural Network Project Guide

Bee Go プロジェクト用の Claude Code (claude.ai/code) ガイダンス

## 🤖 Project Overview

**Bee** (🐝) は、パーセプトロンから大規模言語モデル（LLM）まで段階的にニューラルネットワークを学ぶプロジェクトです。小さなニューロン（Bee）が群れとして協調し、高度な知能を形成します。

**Tech Stack**: Go | Makefile | CLI Tool

## 🚫 CRITICAL: GitHub Operations Restrictions

**最重要ルール: Claude Codeは以下の操作を決して行ってはいけません**

### 🔴 絶対禁止事項
1. **Pull Requestのマージ** - ユーザーのみが判断・実行する
2. **Issueのクローズ** - ユーザーのみが判断・実行する  
3. **ブランチの削除** - ユーザーのみが判断・実行する

### ✅ Claude Codeが実行可能な操作
- リリースの作成
- Pull Requestの作成
- Issueの作成
- ブランチの作成
- コミットの作成とプッシュ
- CI/CDの実行確認（状況報告のみ）

**理由**: これらの操作はプロジェクトの方向性や品質に重大な影響を与えるため、必ずユーザーの明示的な判断と承認が必要です。

## 🔄 Pull Request Creation Rule

**CRITICAL: コード変更後は必ずPull Requestを作成する**

### 必須フロー
1. コード変更完了
2. 品質チェック実行 (`make quality`)
3. 変更をコミット
4. **Pull Request作成** (絶対に忘れてはいけない)
5. ⚠️ **ユーザーによる承認・マージ待ち** (Claude Codeはマージしない)

### PR作成チェックリスト
- [ ] すべてのコード変更が完了している
- [ ] 品質チェックが通っている
- [ ] 適切なブランチ名になっている
- [ ] PR説明が適切に記載されている
- [ ] 関連するIssueが参照されている
- [ ] ユーザーに承認・マージを依頼

## 🎯 Project Architecture

### Phase-Based Learning Structure
```
Phase 1.0: Perceptron    - 線形分類、重み更新
Phase 1.1: MLP          - 多層パーセプトロン、誤差逆伝播
Phase 2.0: CNN/RNN      - 画像処理、系列処理
Phase 3.0: Attention    - Self-Attention、Transformer
Phase 4.0: LLM          - 大規模言語モデル、分散学習
```

### Expected Directory Structure
```
bee/
├── phase1/         # Basic perceptron implementations
├── phase2/         # CNN, RNN implementations  
├── phase3/         # Attention, Transformer implementations
├── phase4/         # LLM, distributed learning
├── cmd/            # CLI tools (bee train / bee infer)
├── datasets/       # Dataset management
├── benchmark/      # Performance comparison tools
├── visualization/  # Visualization tools
├── docs/           # Learning guides
├── Makefile        # Build system
└── go.mod          # Go module definition
```

## 🛠 Essential Commands

```bash
# Development Setup
make setup          # Initial project setup
make dev           # Start development environment
make install       # Install dependencies

# Quality checks
make quality       # Run all quality checks
make quality-fix   # Auto-fix issues
make lint          # Run linting
make format        # Format code
make test          # Run tests

# Build and deployment
make build         # Build for production
make clean         # Clean build artifacts
make benchmark     # Performance benchmarking

# Git workflow
make git-hooks     # Setup pre-commit hooks
make pr-ready      # Prepare for pull request
```

## 🔧 Development Guidelines

### Code Quality Requirements

すべての関数は以下を含む必要があります：

- **Package Documentation**: 各パッケージの目的と使用方法
- **Function Documentation**: 関数の説明、パラメータ、戻り値
- **Error Handling**: 適切なエラーハンドリングパターン
- **Testing**: 単体テストとカバレッジ

### Go Code Style
```go
// Package perceptron implements basic perceptron neural network
package perceptron

import (
    "errors"
    "fmt"
)

// Perceptron represents a basic perceptron neuron
type Perceptron struct {
    weights []float64
    bias    float64
    learning_rate float64
}

// Train trains the perceptron with given input and expected output
func (p *Perceptron) Train(inputs []float64, expected float64) error {
    if len(inputs) != len(p.weights) {
        return errors.New("input size mismatch")
    }
    // Implementation
    return nil
}
```

### Development Workflow

1. **Branch Creation**: 適切な命名規則でブランチ作成
2. **Implementation**: Go のベストプラクティスに従って実装
3. **Testing**: 包括的なテストコード作成
4. **Quality Checks**: `make quality` で品質チェック
5. **Documentation**: コードコメントとドキュメント更新
6. **Pull Request Creation**: 必ずPR作成

### Required for Every Implementation

- **Documentation**: Go docコメントで説明
- **Error Handling**: 適切なエラーハンドリング
- **Testing**: テストカバレッジを保つ
- **Benchmarking**: 性能測定コード

## 🏗 AI-First Design Principles

1. **Type Safety First**: Goの型安全性を最大限活用
2. **Modular Architecture**: 独立したテスト可能なモジュール
3. **Clear Separation of Concerns**: 機能の明確な分離
4. **Predictable Patterns**: 一貫した命名と構造規則
5. **Self-Documenting Code**: 豊富なコメントと型注釈

## 📋 Development Checklist

新機能実装時に確認すること:

- [ ] **Documentation**: Go docコメント記載済み
- [ ] **Testing**: 単体テストとカバレッジ確保
- [ ] **Error Handling**: 適切なエラーハンドリング実装
- [ ] **Performance**: ベンチマークテスト作成
- [ ] **Quality Checks**: `make quality` 通過確認
- [ ] **Pull Request**: PR作成完了

## 🎯 Current State & Implementation Priority

**⚠️ Important**: このリポジトリは現在初期状態で、README.mdのみ存在します。

### Implementation Priority
1. **Phase 1.0**: 基本パーセプトロンから開始
2. **Build System**: Makefile とテスト環境構築
3. **CLI Tool**: `bee` コマンドラインツール実装
4. **Progressive Implementation**: 段階的に高度なアーキテクチャへ

### Performance Goals
- 推論速度: ベースライン比100倍高速化
- 精度向上: パーセプトロン ~70% → LLM ~98%
- 学習効率: 段階的構造による体系的理解

---

**このガイドは効率的なAIエージェント開発と高いコード品質維持を実現します。**
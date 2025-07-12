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

### 🧠 学習重視実装方針

**CRITICAL: ライブラリ依存を避け、アルゴリズム本質理解を最優先**

#### ✅ 許可されるライブラリ使用
```go
// 基本的な数値演算のみ許可
import "gonum.org/v1/gonum/mat"    // 線形代数基本操作
import "gonum.org/v1/gonum/stat"  // 統計計算基本操作
// 低レベル GPU binding（CUDA/OpenCL）
```

#### 🚫 禁止されるライブラリ使用
```go
// 高レベルMLライブラリは禁止
// ❌ import "gorgonia.org/gorgonia"     // TensorFlow like
// ❌ import "github.com/tensorflow/tfgo" // TensorFlow bindings  
// ❌ PyTorch bindings
// ❌ 完成されたニューラルネットワークフレームワーク
```

#### 🎯 段階的実装ルール

1. **ナイーブ実装**: 最も理解しやすい直接実装を必ず最初に作成
2. **最適化実装**: パフォーマンス改善（学習効果と両立）
3. **ライブラリ比較**: 自実装 vs 標準ライブラリの定量比較
4. **理論統合**: 数式→アルゴリズム→コード→テストの完全サイクル

#### 💡 学習効果最大化パターン
```go
// ❌ Bad: ライブラリ丸投げ
func (nn *NeuralNet) Forward(x mat.Matrix) mat.Matrix {
    return someLibrary.Predict(x)  // 学習効果ゼロ
}

// ✅ Good: 段階的理解重視実装
func (nn *NeuralNet) Forward(x []float64) []float64 {
    // Step 1: 重み付き和計算（明示的実装）
    weightedSum := 0.0
    for i, weight := range nn.weights {
        weightedSum += x[i] * weight  // 各計算を明示
    }
    
    // Step 2: バイアス追加
    weightedSum += nn.bias
    
    // Step 3: 活性化関数（自実装）
    return nn.sigmoid(weightedSum)  // 関数内部も自実装
}

// 活性化関数の自実装例
func (nn *NeuralNet) sigmoid(x float64) float64 {
    // 数式: σ(x) = 1 / (1 + e^(-x))
    return 1.0 / (1.0 + math.Exp(-x))
}
```


### Go Code Style（学習重視パターン）
```go
// Package perceptron implements basic perceptron neural network
// Mathematical Foundation: McCulloch-Pitts neuron model (1943)
// Learning Goal: Understanding linear classification and weight updates
package perceptron

import (
    "errors"
    "fmt"
    "math"
)

// Perceptron represents a basic perceptron neuron
// Mathematical Model: y = σ(w·x + b) where σ is activation function
type Perceptron struct {
    weights      []float64  // synaptic weights (w)
    bias         float64    // bias term (b)
    learningRate float64    // learning rate (α)
}

// NewPerceptron creates a new perceptron with random weights
// Learning Rationale: Understanding initialization strategies
func NewPerceptron(inputSize int, learningRate float64) *Perceptron {
    weights := make([]float64, inputSize)
    // Xavier initialization for better convergence
    for i := range weights {
        weights[i] = (rand.Float64()*2 - 1) / math.Sqrt(float64(inputSize))
    }
    
    return &Perceptron{
        weights:      weights,
        bias:         0.0,  // Start with zero bias
        learningRate: learningRate,
    }
}

// Forward performs forward propagation
// Mathematical Foundation: y = σ(Σ(wi * xi) + b)
// Learning Goal: Understanding weighted sum and activation
func (p *Perceptron) Forward(inputs []float64) (float64, error) {
    if len(inputs) != len(p.weights) {
        return 0, errors.New("input size mismatch")
    }
    
    // Step 1: Calculate weighted sum (明示的実装)
    weightedSum := p.bias
    for i, input := range inputs {
        weightedSum += p.weights[i] * input
    }
    
    // Step 2: Apply activation function (Heaviside step function)
    // Mathematical: σ(x) = 1 if x ≥ 0, else 0
    if weightedSum >= 0.0 {
        return 1.0, nil
    }
    return 0.0, nil
}

// Train performs one training iteration using perceptron learning rule
// Mathematical Foundation: Δw = α(t - y)x where t=target, y=output
// Learning Goal: Understanding gradient-free weight updates
func (p *Perceptron) Train(inputs []float64, target float64) error {
    if len(inputs) != len(p.weights) {
        return errors.New("input size mismatch")
    }
    
    // Step 1: Forward propagation
    output, err := p.Forward(inputs)
    if err != nil {
        return err
    }
    
    // Step 2: Calculate error
    error := target - output
    
    // Step 3: Update weights (perceptron learning rule)
    // Mathematical: wi = wi + α * error * xi
    for i, input := range inputs {
        p.weights[i] += p.learningRate * error * input
    }
    
    // Step 4: Update bias
    // Mathematical: b = b + α * error
    p.bias += p.learningRate * error
    
    return nil
}
```

### 🎓 学習重視開発ワークフロー

1. **理論理解**: 実装前に数式・アルゴリズムの数学的背景を理解
2. **ナイーブ実装**: 最も理解しやすい直接実装から開始
3. **テスト作成**: 数値的正確性確認を含む包括的テスト
4. **性能測定**: ベースライン測定と最適化前後の比較
5. **最適化実装**: 学習効果を維持しながら性能改善
6. **ライブラリ比較**: 自実装 vs 標準ライブラリの定量評価
7. **品質チェック**: `make quality` で品質確認
8. **ドキュメント**: 学習観点と数学的背景を含む説明
9. **Pull Request作成**: 必ずPR作成

### 🧪 学習効果検証要件

実装ごとに以下を必ず含む：

#### 必須実装要素
- **Mathematical Foundation**: 実装する数式の詳細説明
- **Step-by-Step Implementation**: 各計算ステップの明示的実装
- **Learning Rationale**: 実装選択の学習観点説明
- **Numerical Validation**: 既知解との数値比較テスト

#### 必須テスト要素
- **Unit Tests**: 各関数の動作確認
- **Integration Tests**: アルゴリズム全体の動作確認
- **Numerical Tests**: 数学的正確性の確認
- **Performance Tests**: 実行時間・メモリ使用量測定
- **Comparison Tests**: 他実装との結果比較

#### 必須ドキュメント要素
- **Algorithm Explanation**: アルゴリズムの動作原理
- **Mathematical Derivation**: 数式の導出過程
- **Implementation Notes**: 実装上の注意点・学習ポイント
- **Performance Analysis**: 性能特性の分析

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
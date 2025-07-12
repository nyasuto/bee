# ARM64 Mac DevContainer セットアップガイド

Apple Silicon Mac（M1/M2/M3）でのBee Neural Network開発環境最適化ガイド

## 🍎 ARM64 Mac特有の課題と解決策

### 課題

1. **TensorFlow ARM64 Linux対応**: `linux/arm64`でのTensorFlowサポートが限定的
2. **CUDA非対応**: ARM Mac上のLinuxコンテナではCUDAが利用不可
3. **性能比較制約**: PyTorch/TensorFlow比較の実行環境制約

### 解決策

**ハイブリッド環境戦略**を採用：
- **DevContainer**: Go開発 + 基本Python環境（PyTorch CPU）
- **ホストMac**: TensorFlow/PyTorch ネイティブ実行
- **クラウド**: 大規模GPU比較実行

## 🚀 ARM64最適化DevContainer使用方法

### クイックスタート

```bash
# 1. プロジェクトクローン
git clone https://github.com/nyasuto/bee.git
cd bee

# 2. ARM64専用DevContainer起動
cp .devcontainer/devcontainer-arm64.json .devcontainer/devcontainer.json
code .
# "Reopen in Container" を選択

# 3. ARM64環境自動セットアップ
# postCreateCommand で make setup-arm64 が自動実行
```

### 手動ARM64セットアップ

```bash
# ARM64専用セットアップ
make setup-arm64

# ARM64環境検証
make verify-arm64

# ARM64ビルド
make build-arm64
```

## 🛠 ARM64環境構成

### DevContainer内環境

#### ✅ 利用可能
- **Go 1.21**: ARM64ネイティブ対応
- **Python 3.11**: 基本的なML環境
- **PyTorch 2.1.0**: CPU版（ARM64対応）
- **NumPy/SciPy/Matplotlib**: 完全対応
- **数値計算ライブラリ**: OpenBLAS, LAPACK（ARM64最適化）

#### ⚠️ 制限あり
- **TensorFlow**: ARM64 Linux対応が不安定
- **CUDA**: ARM Macでは利用不可
- **GPU加速**: Metal Performance Shadersはホスト側のみ

### ホストMac推奨環境

ARM Mac上でML比較を行う場合：

```bash
# ホストMacでのTensorFlow/PyTorch環境
# Python 3.11 with conda or venv
pip install tensorflow-macos tensorflow-metal
pip install torch torchvision torchaudio

# 性能比較実行
python scripts/compare-ml-frameworks.py
```

## 🔧 開発ワークフロー

### パターン1: DevContainer主体開発

```bash
# DevContainer内での開発
code .  # ARM64 DevContainer起動

# Go実装開発
make dev
make quality
bee train --model=perceptron

# 基本的なPyTorch比較（CPU）
make benchmark-pytorch-cpu
```

### パターン2: ハイブリッド開発

```bash
# DevContainer: Go開発
# ホストMac: ML比較

# 1. DevContainer内でGo実装
make dev
bee train --model=perceptron --output=/workspace/models/

# 2. ホストMacでML比較
python scripts/benchmark-host.py --model=/workspace/models/perceptron.model
```

### パターン3: クラウド統合開発

```bash
# 大規模比較はクラウド実行
make benchmark-cloud --provider=colab --gpu=T4
```

## 📊 性能比較戦略

### ARM64 DevContainer内比較

```bash
# 利用可能な比較
make benchmark-go-vs-pytorch-cpu
make benchmark-numpy-vs-gonum
make benchmark-memory-usage
```

### ホスト統合比較

```bash
# ホストMacでの高性能比較
./scripts/host-benchmark.sh
# - TensorFlow Metal Performance Shaders
# - PyTorch MPS backend
# - Go実装との性能比較
```

## 🛡 トラブルシューティング

### 問題1: TensorFlow import エラー

```bash
# DevContainer内
❌ ModuleNotFoundError: No module named 'tensorflow'

# 解決策: ホストMacでTensorFlow使用
✅ make benchmark-host-tensorflow
```

### 問題2: CUDA関連エラー

```bash
# エラー: CUDA not available
# 解決策: CPU版に切り替え
export CUDA_VISIBLE_DEVICES=""
make benchmark-cpu-only
```

### 問題3: 性能が遅い

```bash
# ARM64最適化確認
make verify-arm64

# OpenBLAS設定確認  
export OPENBLAS_NUM_THREADS=8
make benchmark-optimized
```

## 🎯 最適化Tips

### Go開発最適化

```bash
# ARM64専用ビルド
GOOS=darwin GOARCH=arm64 go build -o bin/bee-arm64

# CGO最適化
CGO_ENABLED=1 go build -tags=openblas
```

### Python環境最適化

```bash
# ARM64 Python最適化
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# NumPy最適化確認
python -c "import numpy; numpy.show_config()"
```

### メモリ最適化

```bash
# Docker Desktop設定
# Memory: 8GB以上
# Swap: 2GB以上
# Disk: SSD推奨
```

## 📈 ベンチマーク例

### DevContainer内実行

```bash
# Go vs PyTorch CPU比較
$ make benchmark-arm64
🍎 ARM64 Benchmark Results:
  Go Perceptron:     1.2ms (inference)
  PyTorch CPU:       3.1ms (inference)  
  Memory Usage:      15MB vs 45MB
```

### ホスト統合実行

```bash
# フル機能比較（ホスト実行）
$ ./scripts/benchmark-host-full.sh
🍎 ARM64 Mac Full Benchmark:
  Go Implementation:     1.2ms
  PyTorch (MPS):        0.8ms
  TensorFlow (Metal):   1.1ms
```

## 🔄 従来環境との比較

| 項目 | 従来DevContainer | ARM64最適化 |
|------|------------------|-------------|
| セットアップ | ❌ TensorFlowエラー | ✅ 10分で完了 |
| Go開発 | ✅ 正常動作 | ✅ ARM64最適化 |
| PyTorch | ❌ 不安定 | ✅ CPU版安定動作 |
| TensorFlow | ❌ 動作困難 | ⚠️ ホスト側推奨 |
| GPU加速 | ❌ 不可 | ⚠️ ホストMPS利用 |
| 開発体験 | 🔺 エラー頻発 | ✅ スムーズ |

## 🚀 次のステップ

ARM64最適化DevContainerで効率的な開発を：

1. **Phase 1実装**: DevContainer内でGo開発
2. **基本比較**: PyTorch CPU版での比較検証
3. **高性能比較**: 必要時ホストMac環境活用
4. **Cloud拡張**: 大規模実験はクラウドGPU活用

ARM Mac特有の制約を理解して、実用的な開発環境を構築できます！
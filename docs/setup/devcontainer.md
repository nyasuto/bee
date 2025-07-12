# DevContainer セットアップガイド

Bee Neural Network プロジェクトのDevContainer開発環境セットアップ手順

## 🎯 概要

DevContainerを使用することで、以下の利点があります：

- **環境統一**: 全開発者で完全に同一の開発環境
- **即座のセットアップ**: 複雑な依存関係を自動インストール  
- **Claude Code最適化**: AI Agent駆動開発に最適化された環境
- **学習効果最大化**: Go + Python + GPU環境の統一による正確な比較

## 🚀 クイックスタート

### 前提条件

1. **Docker Desktop** インストール済み
2. **VS Code** インストール済み  
3. **Dev Containers拡張** インストール済み

```bash
# VS Code Dev Containers拡張のインストール
code --install-extension ms-vscode-remote.remote-containers
```

### 起動手順

```bash
# 1. プロジェクトクローン
git clone https://github.com/nyasuto/bee.git
cd bee

# 2. VS CodeでDevContainer起動
code .
# VS Codeが自動的にDevContainer使用を提案
# "Reopen in Container" を選択

# 3. 自動セットアップ完了まで待機
# postCreateCommand で make setup-dev が自動実行
```

## 📁 DevContainer構成

### ディレクトリ構造

```
.devcontainer/
├── devcontainer.json    # DevContainer設定
├── docker-compose.yml   # Docker Compose設定  
├── Dockerfile           # マルチステージビルド
└── cache/              # キャッシュディレクトリ
```

### 設定されている環境

#### Go開発環境
- **Go 1.21**: 最新の安定版
- **golangci-lint**: コード品質チェック
- **goimports**: インポート自動整理
- **delve**: デバッガ
- **VS Code Go拡張**: IntelliSense、デバッグ統合

#### Python ML環境
- **Python 3.11**: 機械学習ライブラリ対応
- **PyTorch 2.1.0**: ニューラルネットワーク比較用
- **TensorFlow 2.13.0**: ベンチマーク比較用
- **NumPy, SciPy, Matplotlib**: 数値計算・可視化
- **Jupyter**: 学習効果検証用

#### 数値計算ライブラリ
- **OpenBLAS**: 線形代数高速化
- **LAPACK**: 数値計算基盤
- **OpenMP**: 並列化ライブラリ
- **FFTW**: 高速フーリエ変換（CNN用）

## 🎮 使用方法

### 基本コマンド

```bash
# 開発環境セットアップ
make setup-dev

# 環境検証
make verify-setup

# 品質チェック
make quality

# クイックテスト
make test-quick
```

### Claude Code統合

DevContainer環境でClaude Codeを使用：

```bash
# Claude Code開始
claude-code

# DevContainer環境でのClaude Code利点：
# ✅ make quality が確実に動作
# ✅ 依存関係エラーがゼロ
# ✅ PyTorch/TensorFlow比較環境準備済み
# ✅ GPU環境（該当時）も自動設定
```

### GPU環境

GPU環境が利用可能な場合：

```bash
# GPU対応コンテナ起動
make docker-gpu

# GPU環境確認
nvidia-smi  # NVIDIA GPU情報表示
```

## 🔧 カスタマイズ

### VS Code設定

DevContainerに含まれるVS Code設定：

```json
{
  "go.lintTool": "golangci-lint",
  "go.formatTool": "goimports",
  "go.testFlags": ["-v", "-race"],
  "editor.formatOnSave": true
}
```

### 環境変数

設定されている環境変数：

```bash
GOPROXY=https://proxy.golang.org,direct
GOSUMDB=sum.golang.org
CGO_ENABLED=1
CLAUDE_PROJECT_TYPE=neural-network
CLAUDE_LANGUAGE=go
CLAUDE_LEARNING_MODE=enabled
```

### Python依存関係追加

追加のPythonライブラリが必要な場合：

```dockerfile
# .devcontainer/Dockerfile の development ステージに追加
RUN pip3 install additional-package==version
```

## 🚨 トラブルシューティング

### よくある問題

#### 1. Docker Desktop未起動
```bash
# エラー: Cannot connect to Docker daemon
# 解決: Docker Desktop を起動
```

#### 2. ポート競合
```bash
# エラー: Port already in use
# 解決: .devcontainer/docker-compose.yml のポート変更
```

#### 3. 権限問題
```bash
# エラー: Permission denied
# 解決: DevContainer再ビルド
code --command "Dev Containers: Rebuild Container"
```

#### 4. メモリ不足
```bash
# エラー: Out of memory
# 解決: Docker Desktop のメモリ設定を増加（8GB以上推奨）
```

### 環境リセット

完全リセットが必要な場合：

```bash
# 1. DevContainerクリーンアップ
make docker-clean

# 2. Dockerイメージ削除
docker rmi bee-dev

# 3. VS CodeでDevContainer再ビルド
code --command "Dev Containers: Rebuild Container"
```

## 📊 パフォーマンス最適化

### ビルドキャッシュ

DevContainerはビルドキャッシュを活用：

```yaml
# docker-compose.yml
volumes:
  - cache-go:/home/vscode/.cache/go-build
  - cache-go-mod:/home/vscode/go/pkg/mod
```

### リソース設定

推奨リソース設定：

```yaml
# Docker Desktop設定
Memory: 8GB以上
CPU: 4コア以上
Disk: 20GB以上の空き容量
```

## 🆚 ローカル環境との比較

| 項目 | ローカル環境 | DevContainer |
|------|-------------|--------------|
| セットアップ時間 | 2-4時間 | 5-10分 |
| 環境一貫性 | OS依存 | 完全統一 |
| 依存関係管理 | 手動 | 自動 |
| Claude Code実行 | 環境エラーリスク | 確実動作 |
| 新規参加者対応 | 複雑 | 即座開始 |

## 🔄 更新・メンテナンス

### DevContainer更新

```bash
# 1. 設定変更後、コンテナ再ビルド
code --command "Dev Containers: Rebuild Container"

# 2. 依存関係更新
make setup-dev
```

### 定期メンテナンス

```bash
# 月次メンテナンス
make docker-clean     # 不要なリソース削除
make verify-setup     # 環境健全性確認
```

これで、Beeプロジェクトでの効率的なDevContainer開発環境が整います！
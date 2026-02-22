# Mechanics 

力学系の数値計算を行うための環境と，実際の数値計算プログラムを一括管理するためのリポジトリ．

# 環境構築

以下は初回に一度だけ行えばよい．

## 必要なツール

- uv: pythonのパッケージマネージャ．仮想環境を作り管理する．
- gfortran: Fortranコンパイラ．gccに含まれる．

## ツールのインストール方法

macの場合：
```
brew install uv gcc
```

## ローカルリポジトリのセットアップ

リポジトリのクローン
```
git clone https://github.com/yoshimura-lab-mbd/mechanics.git
cd mechanics
```
依存パッケージのインストール
```
uv sync
```

# 利用

利用例は `examples/` にある．

## marimo ノートブックの実行

依存を同期した後，次で marimo を起動できる．
```
uv sync
uv run marimo edit examples/double_pendulum.py
```

ブラウザが開いたら，`Run all` で実行する．

## 既存の ipynb ノートブック

これまでの `ipynb` も残してあるので，必要な場合は VSCode + Jupyter 拡張機能で実行できる．

## ドキュメント

https://yoshimura-lab-mbd.github.io/mechanics/ を参照


# 開発

ライブラリ本体は[lib/]()，ライブラリを用いた簡単な計算例は[example/]()におく．具体的な研究用プログラムはプロジェクトごとルート以下にディレクトリを作成する．

## プロジェクトの開発

## ライブラリの開発

push前の確認
```
uv run pyright # 型チェック
uv run pytest # テスト
```

# Mechanics 

力学系の数値計算を行うための環境と，実際の数値計算プログラムを一括管理するためのリポジトリ．

# 環境構築

以下は一度だけ行えばよい．

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

利用例は[example/]()にある．

## ipynb ノートブックの実行

VSCodeの場合，Jupyter拡張機能をインストールした上で，ipynbファイルの実行ボタンを押す．初回実行時にはkernelを選択するようメッセージが出るので，`mechanics ... (.venv/bin/python)`を選択する．

# 開発

ライブラリ本体は[lib/]()，ライブラリを用いた簡単な計算例は[example/]()，具体的な研究用プログラムはプロジェクトごとに分ける．

## プロジェクトの開発

## ライブラリの開発

push前の確認
```
uv run pyright # 型チェック
uv run pytest # テスト
```
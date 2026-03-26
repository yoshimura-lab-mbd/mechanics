## examples

計算例は，marimo向けに書かれている．

- `double_pendulum.py`: 二重振り子のダイナミクス
- `kepler.py`: ケプラー問題

## 図の追加と生成

`examples` 配下の図は `lib/examples/figure/` に置く．

- 図の元ファイルは Asymptote で `*.asy` として書く
- 生成した画像は同じ場所に `*.svg` として置く
- `kepler.py` では `mo.image(src=...)` で SVG を読み込む

生成は例えば次のように行う。

```bash
HOME=/tmp asy -f svg -o lib/examples/figure/kepler_orbit.svg lib/examples/figure/kepler_orbit.asy
```

`asy` が `~/.asy` を作れない環境では，`HOME=/tmp` のように一時的なホームを与えるか，事前に `~/.asy` を作る。

marimoの軌道:
```
uv run marimo run examples/
```

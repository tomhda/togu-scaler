# Your own models / 追加モデル

[日本語は下にあります](#追加モデル日本語)

TOGU SCALER can use 4× super-resolution models that you add yourself, next to the built-in ones.

## Add a model

1. Close TOGU SCALER.
2. Put the model (`.onnx`) into the `models\custom` folder inside the folder that contains `togu-scaler.exe`.
   **Your own models** in the settings (the gear at the top right) shows the folder and opens it.
3. Start the app. The model is at the end of the model lists, marked **(added)**.

## What a model must be

- ONNX format, one input and one output, both float32.
- Input: `1 × 3 × H × W` with fixed numbers (no dynamic sizes), RGB, values from 0 to 1.
- Output: `1 × 3 × 4H × 4W`. Only 4× models are supported.

The input size is used as the tile size: the app cuts the picture into tiles of `W × H`, runs the model on each, and joins the results.
256 or 512 are common choices. Larger tiles need more GPU memory.

Models that do not fit (dynamic sizes, channel-last layout, one channel, float16, 2×) are still listed, and stop with a message when you run them.

## Settings file (optional)

A JSON file with the same name as the model (`mymodel.onnx` → `mymodel.json`) changes how it is listed and run.

```json
{ "name": "My model", "overlap": 16, "video": true }
```

| Key | Meaning | When omitted |
|---|---|---|
| `name` | The name shown in the lists | The file name |
| `overlap` | Pixels of overlap on each side of a tile, 0 to 64. More overlap hides seams and takes longer | 16 |
| `video` | `false` hides the model from the video list | `true` |

A model whose settings file cannot be read is not listed. The reason is shown under **Your own models** in the settings.

## How they run

- On the GPU (DirectML) only. If **Run on** is set to NPU, an added model still runs on the GPU.
- They are not available when **Run on** is set to Vulkan, and they need a DirectX 12 GPU.
- From the command line the key is `custom:<file name without .onnx>`:

```
togu-scaler-cli models --json
togu-scaler-cli run photo.png --model custom:mymodel --json
```

## Making an ONNX file from a PyTorch model

`scripts/npu/export_spandrel.py` in this repository converts models that [spandrel](https://github.com/chaiNNer-org/spandrel) can load
(ESRGAN, Real-ESRGAN Compact, SPAN, SwinIR and others) into a fixed-size ONNX file.
It needs Python with `torch`, `spandrel`, `onnx`, `onnxsim` and `onnxruntime`.

```
python scripts/npu/export_spandrel.py --weights mymodel.pth --name mymodel --tile 256
```

The result is written to `tmp/npu-anime/span/mymodel_nchw_256x256_fp32.onnx`. Copy it into `models\custom`.

## Terms of use

Each model has its own license. Check the terms of a model before you use it; TOGU SCALER does not check them.

---

## 追加モデル（日本語）

TOGU SCALER は、内蔵のモデルに加えて、自分で用意した 4 倍の超解像モデルを使えます。

### モデルを加える

1. TOGU SCALER を閉じる。
2. モデル（`.onnx`）を、`togu-scaler.exe` のあるフォルダの `models\custom` に置く。
   詳細設定（右上の歯車）の「追加モデル」にフォルダの場所が出ていて、そこから開ける。
3. アプリを起動する。モデルの一覧の末尾に「（追加）」付きで並ぶ。

### モデルの条件

- ONNX 形式。入力 1 つ・出力 1 つで、どちらも float32。
- 入力: `1 × 3 × H × W` の固定値（可変サイズは不可）、RGB、値は 0〜1。
- 出力: `1 × 3 × 4H × 4W`。4 倍のモデルのみ。

入力の大きさがタイルの大きさになる。アプリは画像を `W × H` のタイルに切り、1 枚ずつモデルに通して結合する。
256 か 512 が一般的。タイルが大きいほど GPU のメモリを使う。

条件に合わないモデル（可変サイズ、チャンネルが最後の並び、1 チャンネル、float16、2 倍）も一覧には出て、実行したときにメッセージを出して止まる。

### 設定ファイル（任意）

モデルと同じ名前の JSON（`mymodel.onnx` → `mymodel.json`）を置くと、一覧での名前や実行の仕方を変えられる。

```json
{ "name": "My model", "overlap": 16, "video": true }
```

| 項目 | 意味 | 省略したとき |
|---|---|---|
| `name` | 一覧に出す名前 | ファイル名 |
| `overlap` | タイルの各辺の重なり（ピクセル）。0〜64。大きいほど継ぎ目が目立たなくなり、時間がかかる | 16 |
| `video` | `false` にすると動画用の一覧に出さない | `true` |

設定ファイルを読めないモデルは一覧に出ない。理由は詳細設定の「追加モデル」に出る。

### 実行のされ方

- GPU（DirectML）でのみ実行する。「AI実行先」が NPU でも、追加モデルは GPU で実行する。
- 「AI実行先」が Vulkan のときは選べない。DirectX 12 対応の GPU が必要。
- コマンドラインでのキーは `custom:<.onnx を除いたファイル名>`。

```
togu-scaler-cli models --json
togu-scaler-cli run photo.png --model custom:mymodel --json
```

### PyTorch のモデルから ONNX を作る

このリポジトリの `scripts/npu/export_spandrel.py` は、[spandrel](https://github.com/chaiNNer-org/spandrel) が読めるモデル
（ESRGAN、Real-ESRGAN Compact、SPAN、SwinIR など）を固定サイズの ONNX に変換する。
`torch`、`spandrel`、`onnx`、`onnxsim`、`onnxruntime` を入れた Python が必要。

```
python scripts/npu/export_spandrel.py --weights mymodel.pth --name mymodel --tile 256
```

結果は `tmp/npu-anime/span/mymodel_nchw_256x256_fp32.onnx` に出る。これを `models\custom` にコピーする。

### 利用条件

モデルにはそれぞれのライセンスがある。使う前に、そのモデルの利用条件を確認すること。TOGU SCALER は利用条件を確認しない。

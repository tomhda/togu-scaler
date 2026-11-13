# Remaster — 等倍ノイズ除去 / Native-resolution denoising

画像・画像フォルダ・SDR 動画の「処理モデル」で **Remaster（等倍ノイズ除去）** を
選ぶと、解像度を変えずに圧縮ノイズを軽減します。4 倍拡大との連続処理はありません。
既存の「クイック確認」「範囲を選択してクイック確認」で、元画像と比較できます。
出力名は `photo_Remaster_denoised.png` / `clip_Remaster_denoised.mp4` です。

- GPU（DirectML）で実行。NPU を選んでいても Remaster は GPU で実行します。
- Vulkan / SwinIR CUDA には対応しません。失敗時に別の拡大モデルで代替しません。
- 動画の 4K 出力上限は適用せず、元の寸法を維持します。エンコーダがその寸法を
  扱えない場合はエラーになります。奇数寸法の動画はパディングせず、libx264 の
  YUV 4:4:4 で保存します（通常の偶数寸法は YUV 4:2:0）。
- 単一フレームの処理です。動画のちらつきを抑える時間方向の処理はありません。
- 処理後に、1 コマ全体の RGB の平均と標準偏差を入力に合わせ直します（本家の
  `--color-transfer` と同じ処理）。暗い場面で明るさがずれるのを防ぎます。
- 圧縮ノイズの除去と細部の補正を学習したモデルです。細部や粒子感が変わることが
  あるので、先にクイック確認を使ってください。HDR 動画には対応しません。

## ソース版への導入 / Preparing the source checkout

通常の `setup.ps1` で GPU ヘルパーを導入してから、リポジトリのルートで実行:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-remaster.txt
.venv\Scripts\python.exe scripts/get_remaster.py
```

スクリプトは公開 ONNX の固定リビジョンを取得し SHA-256 を検証します。
float16・可変サイズのグラフを、既存ヘルパー用の float32・固定 512×512 に変換します。
学習済み重みの再学習はしません。推論は片側 64px の重なりを持つタイルで行います。
モデルとライセンス・帰属文書を `models/ai/` に置きます。
`onnx` は変換時だけ必要で、アプリの実行時には必要ありません。

`scripts/build_portable.ps1 -WithHelper` はモデルとライセンス文書を同梱します。
モデル未導入の場合はビルドを止めます。アプリの実行中にモデルを取得しません。

## CLI

```powershell
.venv\Scripts\python.exe -m app.cli run photo.png --model Remaster --json
.venv\Scripts\python.exe -m app.cli run clip.mp4 --model Remaster --json
.venv\Scripts\python.exe -m app.cli quick-check clip.mp4 --time 5 --model Remaster --out check.png --json
```

`--scale` を省略すると Remaster は 1、それ以外は従来どおり 4 です。
Remaster に `--scale 2` / `4` を指定すると実行前にエラーになります。
`models --json` の既存の `upscale` 配列に `Remaster` が入り、`scales: [1]` を返します。

Select **Remaster (denoise, original size)** as the image/video processing model.
Images, image folders, and SDR videos retain their source dimensions. Quick check
and selected-area comparison use the same inference path as the full run. This
mode runs on DirectML even when NPU is selected, and never substitutes an upscaler
on failure. It does not chain denoising with 4× super-resolution. It processes each
frame independently, so temporal stability needs to be checked on moving footage.
After inference, each frame's per-channel RGB mean and standard deviation are
matched to the input (the upstream `--color-transfer` step), which keeps dark
scenes from shifting in brightness.
Odd video dimensions use software H.264 YUV 4:4:4 to avoid padding. The normal 4K
output cap is bypassed; unsupported encoder sizes fail instead of being resized.

## Sources

- [Remaster](https://github.com/seantempesta/remaster)
- [Model and model card](https://huggingface.co/seantempesta/remaster-drunet)
- Pinned revision: `018e7815aa8ef6e3eb6433d2572433d4f36e180e`
- Source ONNX SHA-256: `b366ac68ed8e30a5d5a4db7776219ea3bd72929a81a53ef3b6f7e8fd0703206a`
- MIT, Copyright (c) 2026 Sean Tempesta. See
  `scripts/release-licenses/LICENSE-MIT-Remaster.txt` and `NOTICE-Remaster.txt`.

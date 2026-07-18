# llama.cpp b9982 採用評価

## 結論

- 選定ブランチ: `codex/llama-cpp-version-selection`
- 採用crate: `llama-cpp-4 = 0.4.2`
- 同梱 llama.cpp: `b9982` (`99f3dc32296f825fec94f202da1e9fede1e78cf9`)
- 判定: **ブランチへ採用し、3種類のRelease EXEと実モデルスモークテストに合格**

`llama-cpp-4 0.4.2` を製品依存へ切り替え、現行のモデルプリロード、入力停止後の先読み、セッション状態複製、入力評価用と生成用の個別スレッド設定を互換層で維持した。Windows向けビルド不具合はローカルpatchで修正し、`llama-cpp-sys-4` と同梱 llama.cpp をリポジトリ内に固定した。

## 採用構成

| 項目 | 現行値 |
| --- | --- |
| Rust crate | `llama-cpp-4 0.4.2` |
| llama.cpp | b9982 (`99f3dc32296f825fec94f202da1e9fede1e78cf9`) |
| CPU engine | `compatible` / `avx2` / `avx512` |
| GPU / RPC | 無効 |
| link | 静的 |

移行前は `llama_cpp 0.3.2` と2024-04-03時点の llama.cpp commit `60cdf40cc32f0ad4cb11e0ca8fd38f3b93d8d640` を使用していた。

## 候補の選定理由

### `llama-cpp-4 0.4.2` / b9982

- 2026-07-13 公開で、調査時点の llama.cpp に近い。
- ライセンスは MIT OR Apache-2.0。
- llama.cpp の commit を固定して同梱するため、製品ビルドの再現性を管理しやすい。
- 既存実装の主要機能を移植できる Rust API がある。
- `default-features = false` により、不要な動的リンク、RPC、GPU 機能を有効にせず評価できる。

採用した依存指定は次のとおり。

```toml
llama-cpp-4 = { version = "=0.4.2", default-features = false, features = ["openmp"] }
```

### 最新 b10042 を直接選ばない理由

調査時点の最新公式リリースは `b10042` だが、同日に複数リリースされる更新頻度であり、公開直後の版を製品基準に固定する利点は小さい。Rust バインディングが固定している b9982 を最初の移行候補とし、ビルドと回帰試験を通した版を採用する。

### 除外した候補

| 候補 | 判定 | 理由 |
| --- | --- | --- |
| `llama_cpp 0.3.2` 継続 | 除外 | upstream が 2024-04 時点で古い |
| `rs-llama-cpp 0.1.67` | 除外 | 同梱コードが旧 `examples/main` 世代で更新目的を満たさない |
| 公式 b10000 DLL の直接利用 | 保留 | 新世代の互換性確認には有効だが、現行 Rust API からの移行量と DLL 配布管理が増える |

## 公式 b10000 での事前検証

b9982 と近い新世代の互換性確認として、公式 `llama-b10000-bin-win-cpu-x64.zip` を使用した。これは b9982 自体の性能値ではなく、更新後系列の事前検証値として扱う。

- 公式 SHA-256: `1398427465fe3634cc29778b82b8ba990e47a442ef8453ae9d2d114ec2ad6563`
- commit: `47a39665e7081dc482feec169961acc09750a5c4`
- 実モデル: `sarashina2.2-3b-instruct-v0.1-Q4_K_S.gguf`
- CPU: AMD Ryzen 7 7800X3D
- GPU offload: 無効 (`-ngl 0`)
- 結果: モデル読込成功、Zen4 CPU backend 自動選択成功

### `llama-bench` 結果

各条件 3 回。`pp512` は入力評価、`tg64` は生成の参考値。

| thread | pp512 | tg64 |
| ---: | ---: | ---: |
| 4 | 124.809 t/s | 22.830 t/s |
| 5 | 146.944 t/s | 22.507 t/s |
| 6 | 168.485 t/s | 18.808 t/s |

入力評価は 6 thread、生成は 4 thread が最良だった。TrimPrompt が現在進めている入力評価用と生成用の個別スレッド実測には妥当性がある。ただし、アプリ全体の圧縮時間や品質向上を直接示す値ではない。

## 解消したブロッカー

最小 Rust プローブで `llama-cpp-4 0.4.2` の Windows 静的ビルドを試したところ、`llama-cpp-sys-4` のビルド処理が次の形式で CMake を実行した。

```text
cmake --build <build-dir> --parallel 8 -- -j8
```

Visual Studio generator では末尾の `-j8` が MSBuild に渡り、`MSB1001: 不明なスイッチです` で失敗する。手動の CMake build では llama.cpp 本体をコンパイルできたため、主原因は upstream 本体ではなく crate のビルド連携にある。

ローカルpatchでは、Ninja以外に余分な `-j` を渡さず、MSVCのRelease指定を `/O2 /DNDEBUG` に変更した。また、高水準crateが参照するfit、memory、MTP shimのシンボルを解決するため、RPCやMTP機能を有効にせず `llama-common` と必要なshimだけを静的リンクした。

## 依存関係と安全性

- crate は公開直後で利用実績が少なく、現行 crate より更新性は高いが成熟度は低い。
- `build.rs` が CMake、bindgen、同梱 C/C++ コードを実行するため、通常の Rust 依存追加よりビルド時の確認範囲が広い。
- build dependency に `ureq`、`tar`、`flate2` などが入り、依存数が増える。製品ビルドでは取得元とネットワークアクセスの有無を固定・確認する。
- RPC backend は有効化・同梱しない。過去に RPC backend の RCE advisory (`GHSA-j8rj-fmpv-wcxw`) があるため、ローカル圧縮アプリに不要な待受機能を持たせない。
- GPU backend は今回の対象外とし、有効化しない。
- GGUF モデルの Hugging Face 配布元固定と SHA-256 検証は維持する。

## CPUビルド

| engine | AVX | AVX2 | FMA/F16C/BMI2 | AVX-512 | VBMI/VNNI/BF16 | native |
| --- | --- | --- | --- | --- | --- | --- |
| compatible | OFF | OFF | OFF | OFF | OFF | OFF |
| avx2 | ON | ON | ON | OFF | OFF | OFF |
| avx512 | ON | ON | ON | ON | OFF | OFF |

各featureを別target directoryでビルドし、CMakeCacheで上表と `/O2 /DNDEBUG` を確認した。crateとアプリのCPU engine名が不一致の場合は、モデル読込前の自己検査で停止する。

## 検証結果

- compatible: coreテスト143件成功、1件はネットワーク必須のためignore
- AVX2: CPU engine自己検査を実行して成功
- AVX-512: 非対応CPU上でテストEXEのコンパイル・静的リンクまで成功。実行はしていない
- Release: `TrimPrompt-compatible.exe`、`TrimPrompt-avx2.exe`、`TrimPrompt-avx512.exe` の生成成功
- 実モデル: Sarashina 2.2 3B Q4_K_SをAVX2版で読込・圧縮成功
- スモークテスト: 終了コード0、`internal_llm` / `llama_cpp_embedded`、原文フォールバックなし、103文字から79文字へ圧縮
- モデル保持: `-Clean` 更新後も既存GGUFを保持

## 残る受入試験

1. 同一入力・同一設定で旧版と新版の圧縮時間、入力評価速度、生成速度、ピークメモリを比較する。
2. 実プロンプト5件で停止条件、文字化け、出力差を確認する。
3. 基準30件を実LLMで再評価し、重要情報欠落と指示改変を増やしていないことを確認する。
4. AVX-512対応CPUで自己検査と実モデル圧縮を実行する。
5. `cargo test --workspace` と依存監査を実行する。現在の環境には `cargo-audit` がない。

# Prompt Compressor

Prompt Compressor は、Codex 向けの長い依頼文を、必要な条件を保ったまま短くするローカルファーストのプロンプト圧縮アプリです。

アプリ本体は Windows ネイティブアプリ化を前提にしており、最終的には exe だけを起動して文章の圧縮まで完了できる構成を目指しています。

各フォルダの役割は `資料/フォルダ構成.md` にまとめています。

現在の作業状況と他の Codex への引き継ぎ情報は、先に次のファイルを読んでください。

```text
資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md
```

このファイルには、レベル2圧縮改善の目的、実LLM評価結果、残っている失敗、次に実行するコマンドを記録しています。

## 現在の状態

このプロジェクトは Rust workspace として構成されています。

主な構成は次のとおりです。

- 圧縮ロジックを持つ共通 core ライブラリ
- コマンドライン版の圧縮ツール
- 開発確認用のローカル Web UI
- WebView2 を使う Windows デスクトップシェル
- MCP サーバーの土台
- 同梱ローカルモデル、組み込みランタイム、圧縮ポリシー用の設定
- 企画資料、評価メモ、将来構想などの資料

標準の実行経路は `internal_llm` です。

`internal_llm` は、Rust アプリに組み込んだ `llama.cpp` バックエンドと、`application/local/models/` に置いた Sarashina の GGUF モデルを使います。

通常利用では次のものは不要です。

- LM Studio
- `llama.exe`
- `llama-server.exe`
- ローカル LLM 用の HTTP ポート
- アプリ起動時の外部ランタイムダウンロード

LM Studio は、ユーザーが任意のローカルモデルを試すための追加経路として残しています。標準の圧縮処理には不要です。

## フォルダ構成

```text
application/
  core/                           圧縮ロジックとランタイム実装
  ui/                             ローカル Web UI と Windows デスクトップシェル
  interfaces/                     CLI と MCP サーバー
  config/                         Git 管理するモデル・プロファイル・ポリシー設定
  resources/                      LLM テンプレート、JSON スキーマ、評価 fixture
  local/                          PC ごとのモデル、ログ、状態
資料/
  企画資料/                       企画、設計、要件、進捗、作業リスト
  評価メモ/                       評価設計、評価結果、Codex引き継ぎ
  将来構想/                       未実装機能の構想資料
  フォルダ構成.md                 各フォルダの詳細説明
```

## 最初に確認するコマンド

プロファイル一覧を確認します。

```powershell
cargo run -p prompt-compressor-cli -- --list-profiles
```

CLI で圧縮を試します。

```powershell
cargo run -p prompt-compressor-cli -- --profile internal_llm "React の検索ボタンを押したときだけ API を呼ぶように修正してください。URL クエリの状態は維持してください。"
```

開発用 Web UI を起動します。

```powershell
cargo run -p prompt-compressor-local-ui -- --host 127.0.0.1 --port 8787
```

起動後、次を開きます。

```text
http://127.0.0.1:8787
```

Windows デスクトップシェルを起動します。

```powershell
cargo run -p prompt-compressor-desktop
```

## Windows デスクトップ版

デスクトップシェルは、通常の Web サーバーを起動しません。

UI は WebView2 上で表示しますが、HTTP ポートは使わず、`prompt-compressor://` の内部プロトコルとアプリ内 bridge で API を処理します。

そのため、exe 化した後はブラウザで動く Web アプリではなく、通常の Windows アプリとして扱う方針です。

## exe 出力

分離した実行用フォルダを作るには、次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean
```

標準では、ソースフォルダとは別の次の場所へ出力します。

```text
../prompt-compressor-project-exe/
```

パッケージには次が入ります。

- `PromptCompressor.exe`
- `application/config/`
- `application/resources/`
- 設定済みのローカル GGUF モデル

ランタイムは exe に組み込まれるため、`application/local/runtimes/` のような外部ランタイムバイナリはコピーしません。

ただし GGUF モデルは大きいため、exe 本体へ直接埋め込まず、`application/` フォルダ内に配置します。

別の場所へ出力したい場合は、次のように指定します。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean -OutputPath "C:\path\to\PromptCompressor"
```

## ビルドに必要なもの

実行時は自己完結を目指していますが、開発環境で `llama.cpp` 組み込みバックエンドをビルドするには LLVM が必要です。

ビルド時には次のパスを使います。

```text
C:\Program Files\LLVM\bin
```

必要に応じて次の環境変数を設定します。

```powershell
$env:NM_PATH='C:\Program Files\LLVM\bin\llvm-nm.exe'
$env:OBJCOPY_PATH='C:\Program Files\LLVM\bin\llvm-objcopy.exe'
```

## 設定の保存

UI 設定は次に保存します。

```text
application/local/state/ui-settings.json
```

保存対象は、現在の方針では次のようなアプリ設定です。

- 選択中のプロファイル
- 圧縮レベル
- テーマ

入力本文や圧縮結果は保存しない方針です。

## トークン削減の計算ロジック

圧縮結果には、トークン推定と文字数を `入力 → 出力` の形式で表示します。

表示する比率は、圧縮後が圧縮前の何パーセントかを表します。

```text
トークン比 = 出力トークン推定 / 入力トークン推定 × 100
文字数比   = 出力文字数 / 入力文字数 × 100
削減率     = 100 - 比率
```

たとえばトークン比が `78%` の場合、推定トークン数は約 `22%` 削減されています。

ランタイム失敗などで原文を返す場合は、入力と出力が同じになるため比率は `100%` になります。

### トークン推定

現在は、モデル固有の厳密な tokenizer ではなく、ローカルで高速に動く共通の概算を使っています。

日本語が空白で区切られないために `1` トークンと誤表示されないよう、連続する文字種ごとに次のように加算します。

- 日本語文字列: `ceil(文字数 × 2 / 3)`
- 英字と `_` の連続文字列: `ceil(文字数 / 4)`
- 数字の連続文字列: `ceil(文字数 / 3)`
- 空白以外の記号: 1文字ごとに1

実際の消費トークン数は、選択したモデルと tokenizer によって異なります。

## 内蔵ローカル LLM

`internal_llm` プロファイルは、同梱ローカルモデルを組み込み `llama.cpp` バックエンドで直接実行します。

モデルは次の配下に置きます。

```text
application/local/models/
```

現在の主な採用モデル:

```text
sarashina2.2-3b-instruct
```

CLI で `internal_llm` を使う例:

```powershell
cargo run -p prompt-compressor-cli -- --profile internal_llm "この依頼文を、条件を落とさず短くしてください。"
```

モデルが見つからない、またはローカルランタイムが失敗した場合は、危険な圧縮を避けるため原文を返し、JSON 出力では `should_send_original=true` を付けます。

## LM Studio 接続

`lmstudio_local` プロファイルは、ユーザーが LM Studio で読み込んだ任意のローカルモデルを試すために残しています。

利用手順:

1. LM Studio を起動する。
2. `http://127.0.0.1:1234/v1` のローカルサーバーを有効にする。
3. 試したいモデルを LM Studio で読み込む。
4. UI で `LM Studio（ローカルモデル自由選択）` を選ぶ、または CLI で `--profile lmstudio_local` を指定する。

この経路は比較・検証用です。

標準の exe 利用では LM Studio は不要です。

## プロンプト評価

レベル2の圧縮品質は、次の fixture で評価しています。

```text
application/resources/evaluations/raw-prompts-level2-evaluation-v1.json
```

この評価では、実際のアプリと同じ LLM 経路を必ず使います。

禁止すること:

- モック結果で合格扱いにする
- 手書きの期待出力だけで合格扱いにする
- LM Studio 経路で `internal_llm` の代わりにする

必須プロファイル:

```text
internal_llm
```

単体テストとCLIビルド:

```powershell
$env:NM_PATH='C:\Program Files\LLVM\bin\llvm-nm.exe'
$env:OBJCOPY_PATH='C:\Program Files\LLVM\bin\llvm-objcopy.exe'
cargo test -p prompt-compressor-core -p prompt-compressor-cli --no-default-features
cargo build -p prompt-compressor-cli
```

特定ケースだけを実LLMで再評価する例:

```powershell
target/debug/prompt-compressor-cli.exe --profile internal_llm --eval-fixture application/resources/evaluations/raw-prompts-level2-evaluation-v1.json --eval-levels 2 --eval-case-offset 23 --eval-case-limit 1 --eval-progress
```

全30件を1件目から実行する場合:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1
```

失敗ケースを直した後は、最後に必ず全30件を1件目から再実行します。

完了条件:

```text
passed=true
case_count=30
run_count=30
failure_count=0
```

## 現在の未完了作業

レベル2圧縮改善は作業中です。

直近の実LLM個別再チェックでは、次は通過済みです。

- `normal-18`
- `normal-19`
- `normal-20`

残っている失敗:

- `unexpected-04`: CSV インポートで「10MB超過時は読み込まず」が落ちている
- `unexpected-05`: Next.js order API で「入力チェック/入力検証」と「テスト」が落ちている

詳細は次を参照してください。

```text
資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md
```

## 次にやること

1. `unexpected-04` と `unexpected-05` の圧縮候補を修正する。
2. 単体テストと CLI ビルドを再実行する。
3. `internal_llm` で失敗ケースを個別再評価する。
4. 個別再評価が通ったら、全30件を1件目から再実行する。
5. 全件合格したら、評価結果を `資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md` に追記する。

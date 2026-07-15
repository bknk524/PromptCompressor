# TrimPrompt

TrimPrompt は、Codex 向けの長い依頼文を、必要な条件を保ったまま短くするローカルファーストのプロンプト圧縮アプリです。完成品は Windows EXE を中心とし、Web版は開発確認用として扱います。

Windowsデスクトップ版は、WebView2、組み込み `llama.cpp`、ローカルGGUFモデルをアプリ内で連携させます。通常利用時にLM Studioや外部LLMサーバーは不要です。

各フォルダの役割は `資料/フォルダ構成.md` にまとめています。

レベル2評価の現在状態と再現手順は、先に次のファイルを確認してください。

```text
資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md
```

このファイルには、評価条件、直近の実LLM全件結果、個別・全件評価コマンドを記録しています。

## 現在の状態

最終更新: 2026-07-15

このプロジェクトは Rust workspace として構成されています。

製品名と配布EXEは `TrimPrompt` に統一しています。既存環境との互換性を保つため、Rustクレート名、環境変数、WebView2の内部プロトコルには従来の `prompt-compressor` 識別子を残しています。

主な構成は次のとおりです。

- 圧縮ロジックを持つ共通 core ライブラリ
- コマンドライン版の圧縮ツール
- 開発確認用のローカル Web UI
- WebView2 を使う Windows デスクトップシェル
- ユーザー操作で初回取得するローカルモデル、組み込みランタイム、圧縮ポリシー用の設定
- 企画資料、評価メモ、将来構想などの資料

標準の実行経路は `internal_llm` です。

`internal_llm` は、Rustアプリに組み込んだ `llama.cpp` バックエンドと、`application/local/models/` に置くSarashinaのGGUFモデルを使います。モデルがない場合は、UIの「モデルを取得」からHugging Faceの固定リビジョンを取得します。ダウンロードは中止・再開でき、完了時にファイルサイズとSHA-256を検証します。

現在の動作方針:

- デスクトップ版と開発用Web UIの圧縮モードは「標準」と「高圧縮」の2択。標準は既存のレベル2、高圧縮は既存のレベル3へ対応し、今回のUI整理では圧縮プロンプト、前処理、後処理、検証条件を変更していない
- 保存済みUI設定が旧レベル1の場合は、読み込み時に標準へ移行する
- core、CLI、評価ツールは引き続きレベル0〜3を扱う。レベル1、2、3の新規入力は、前処理だけで結果を確定せず、必ず選択中のLLM経路を実行する
- レベル0の原文維持だけはLLMを実行しない
- ローカル推論はCPUで実行し、GPU推論は有効にしない
- CPUスレッド数は自動調整し、アプリ操作用の余力を残したうえで最大8スレッドを使う
- モデル取得、モデル読み込み、プロンプト準備、文章圧縮はUIスレッド外で処理する
- モデルと圧縮プロンプトを再利用し、2回目以降の推論開始を短縮する
- 同じ原文、レベル、プロファイル、制約を再実行した場合だけ、正常な実LLM出力を最大16件のメモリ内LRUから再利用する。入力と結果はディスクへ保存せず、アプリ終了時に破棄する
- 実LLM評価は結果LRUを使用せず、同じ入力の再評価でも毎回モデル推論を実行する
- LLM出力は必須語、数値、否定、対象限定、状態維持、検証条件などを確認し、不正な場合は原文を返す
- EXE版は1プロセスだけ起動し、2回目以降は既存ウィンドウを前面へ戻して「すでに起動しています」と表示する
- EXE版はHTTPサーバーを起動せず、WebView2の内部プロトコルとアプリ内bridgeで通信する
- 開発中のサンプル文章は入力欄へ本文を入れるだけで、モデル、圧縮モード、保存設定、圧縮結果を変更しない。完成版ではサンプル用スクリプトとUIを削除する

UIは既定のデスクトップサイズでページ全体を固定し、入力欄と結果欄だけを必要に応じて内部スクロールさせます。タイトル横には現在のモデル種別と圧縮モードを小枠で表示し、設定画面では「標準」と「高圧縮」をセグメント式で切り替えます。入力欄の見出し、サンプル選択、クリア、圧縮ボタンはデスクトップ版の最小幅でも一列を維持し、それ以外の領域はウィンドウ幅に合わせて再配置します。

### 現在の検証結果

2026-07-15時点の最新ソースとリリースEXEで、次を確認済みです。

- `cargo fmt --all -- --check`: 成功
- `cargo test --workspace --all-features`: 169件成功、Hugging Face通信を伴う1件を除外
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: 成功
- リリースビルド: 成功
- パッケージ内の実モデル圧縮スモークテスト: 成功

最新の `TrimPrompt.exe` は `../TrimPrompt-exe/` に生成済みです。

### 現在の未確認事項

- 2026-07-13のレベル2基準30件は全件合格している
- その後に前処理、モデル入力、CPU推論設定を変更したため、最新実装に対する30件全件評価は再実行が必要
- 配布方法、インストーラー、コード署名は現段階の対象外

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
  interfaces/                     CLI
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

UI は WebView2 上で表示しますが、HTTP ポートは使わず、互換性維持用の `prompt-compressor://` 内部プロトコルとアプリ内 bridge で API を処理します。この内部識別子は画面には表示しません。

そのため、exe 化した後はブラウザで動く Web アプリではなく、通常の Windows アプリとして扱う方針です。

## exe 出力

分離した実行用フォルダを作るには、次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean
```

標準では、ソースフォルダとは別の次の場所へ出力します。

```text
../TrimPrompt-exe/
```

パッケージには次が入ります。

- `TrimPrompt.exe`
- `application/config/`
- `application/resources/prompts/`
- モデル、キャッシュ、ログ、状態を保存する `application/local/`（初回生成時は空）

ランタイムは exe に組み込まれるため、`application/local/runtimes/` のような外部ランタイムバイナリはコピーしません。

GGUFモデルは約1.97GBあるためパッケージへ同梱せず、初回セットアップ時にHugging Faceから取得して `application/local/models/` に配置します。

同じ出力先へ `-Clean` を付けて更新しても、`application/local/` と `application/config/user/` は削除しません。取得済みモデル、途中まで取得したモデル、設定、キャッシュ、ログ、WebView2状態を保持したまま、EXEとアプリ管理ファイルだけを更新します。既存モデルを使ったパッケージ内スモークテストでも、モデルを上書き・削除しません。

別の場所へ出力したい場合は、次のように指定します。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean -OutputPath "C:\path\to\TrimPrompt"
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
- 圧縮モード（標準または高圧縮。内部値はレベル2または3）
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

`internal_llm` プロファイルは、UIからHugging Faceより取得したローカルモデルを組み込み `llama.cpp` バックエンドで直接実行します。取得元のコミット、ファイルサイズ、SHA-256は `application/config/model-catalog.yaml` に固定されています。

モデルは次の配下へ自動配置されます。

```text
application/local/models/
```

現在の主な採用モデル:

```text
sarashina2.2-3b-instruct-v0.1 Q4_K_S
```

CLI で `internal_llm` を使う例:

```powershell
cargo run -p prompt-compressor-cli -- --profile internal_llm "この依頼文を、条件を落とさず短くしてください。"
```

モデルの取得・検証またはローカルランタイムが失敗した場合は、危険な圧縮を避けるため原文を返し、JSON 出力では `should_send_original=true` を付けます。

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

実LLM評価は、実際のアプリと同じ `internal_llm` プロファイルと `llama_cpp_embedded` ランタイムを使います。モック、手書き出力、LM Studioの結果を代用して合格扱いにはしません。

現在の主要fixture:

- `raw-prompts-level2-evaluation-v1.json`: レベル2の基準30件（通常20件、想定外入力10件）
- `raw-long-redundant-evaluation-v1.json`: 長文・冗長入力25件（通常20件、誤字やノイズを含む入力5件）をレベル1、2、3で評価
- `level-profile-evaluation-v1.json`: レベル別プロファイルの評価

```text
application/resources/evaluations/raw-prompts-level2-evaluation-v1.json
application/resources/evaluations/raw-long-redundant-evaluation-v1.json
application/resources/evaluations/level-profile-evaluation-v1.json
```

レベル2の特定ケースだけを実行する例（`--eval-case-offset` と同じく0始まり）:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1 -RunId unexpected-04-recheck -CaseOffset 23 -CaseLimit 1
```

レベル2の基準30件を1件目から実行:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/run-raw-level2-real-llm-eval.ps1
```

長文・冗長入力25件を全レベルで実行:

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/evaluate-prompt-profiles.ps1 -FixturePath application/resources/evaluations/raw-long-redundant-evaluation-v1.json -Levels 1,2,3
```

評価スクリプトは必要に応じてCLIをビルドし、`NM_PATH`、`OBJCOPY_PATH`、`LIBCLANG_PATH` を既存環境、LLVM、Rustツールチェーンから解決します。`CARGO_TARGET_DIR` が設定されている場合はその出力先を使います。

各レベルは平均圧縮率に加え、`min_case_character_ratio` と `max_case_character_ratio` でケース単位の下限・上限も検証します。下限は必要情報を失うほどの過圧縮、上限は圧縮不足を検出します。境界値は合格として扱い、範囲外または下限が上限を超えるfixtureは実行前にエラーになります。

レベル2専用スクリプトの結果は次へ保存されます。

```text
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.json
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.progress.log
application/resources/evaluations/results/raw-level2-real-llm-<RunId>.status.json
```

レベル2基準の完了条件:

```text
passed=true
case_count=30
run_count=30
failure_count=0
```

2026-07-13の実LLM全件評価では、当時の実装でレベル2基準30件がこの条件を満たしました。その後に前処理、モデル入力、CPU推論設定を変更しているため、現在の実装を品質確定するには全30件の再実行が必要です。圧縮プロンプト、前処理、後処理、検証、fixture、モデルのいずれかを変更した場合は、関連ケースを個別確認した後、必ず全30件を再実行します。

評価の詳細と引き継ぎ情報:

```text
資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md
```

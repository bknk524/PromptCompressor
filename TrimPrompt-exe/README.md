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

最終更新: 2026-07-22

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
- `TrimPrompt.exe` は起動直後にCPU命令セットをOS込みで判定する。SSE4.2、AVX2、FMA、F16C、BMI2を満たせば安全な初期値としてAVX2版を選び、SSE4.2だけを満たす場合は互換版へ戻す。SSE4.2非対応CPUでは安全に実行できる内部EXEがないため、案内を表示して起動しない。AVX-512版はAVX512F、CD、BW、DQ、VLもすべて満たすCPUだけを候補にし、初回から無条件には使わない
- モデル取得後の初回起動では通常UIをまだ表示せず、リング、「初期設定中」、「完了まで少し時間がかかります」の文言を表示してCPU診断を行う。CPUスレッド数は実モデルで128トークンの入力評価と12トークンの逐次生成を交互に3回ずつ計測し、ばらつきが20%を超えた候補は2回追加測定する。高スレッド候補だけでなく2・4スレッドも測る。続いて512トークンの長文入力評価で物理マイクロバッチ128・256・512を各3回測る。既定の512より3%以上速い候補が出た場合だけ、実際の圧縮プロンプト3種を両方の幅で最大192トークン生成し、生出力がすべて一致した候補だけを採用する。1件でも違えば512を維持する。結果をCPU命令セット、推論エンジン、CPU、モデル、実行設定ごとに `application/local/state/inference-tuning-v1/` へ保存する
- AVX-512対応CPUでは、スレッド測定後にAVX2版とAVX-512版を5種類の固定入力による実圧縮パイプラインで比較する。代替版の子プロセスは同じ初期設定中にその版のスレッド診断も完了する。各出力が原文と一致するかではなく、ケースごとの必須要件、数値、禁止条件、検証条件を満たすかを判定する。品質を満たす版同士では、前処理、入力準備、LLM生成、検証までの平均時間を比べ、AVX-512版が3%以上速い場合だけ次回起動から採用する。AVX-512版だけが品質条件を外した場合は速度にかかわらずAVX2版を維持する
- 診断結果を保存したらランチャー経由で一度だけ自動再起動し、選択したCPU版とスレッド設定を読み込む。以後は保存設定が有効な限り診断画面を出さず、最適化済み設定で通常UIを起動する
- 診断に失敗した場合は未最適化の通常UIへ進まず、初期設定画面にエラーを表示する。次回起動時に保存状態を再確認し、未完了の診断を再試行する
- CPUエンジンの選択結果はCPU識別情報と、推論コア、モデル設定、プロンプト、llama.cppをまとめた推論互換IDに結び付ける。UIだけを更新したビルドでは測定結果を再利用し、CPU、モデル、診断方式、推論実装が変わった場合だけ初期設定画面で再測定する。実行ファイルの取り違え防止には別のビルドIDを使い、選択した内部EXEが欠けている場合はAVX2版または互換版へ戻す。設定画面の「CPU最適化」から手動で再調整した場合も同じ初期設定画面と自動再起動を使う
- 設定の「詳細設定」では、CPU命令セットを自動、SSE4.2互換、AVX2、AVX-512から選択し、生成スレッド数と入力評価スレッド数を自動または手動で指定できる。非対応のCPU版は選択不可とし、手動スレッド数は現在の論理CPU数を上限に検証する。実際に稼働中のCPU版と両スレッド数を同じ画面に表示し、変更はランチャー経由の再起動後に反映する
- 手動指定は自動診断の保存結果を削除せずに上書きする。「自動」へ戻した場合は、CPU・モデル・推論互換IDに適合する測定済み設定を再利用する
- モデル取得、モデル読み込み、プロンプト準備、文章圧縮はUIスレッド外で処理する
- 組み込みLLMのタイムアウトはモデル取得、セッション準備、入力評価、生成を含む共通期限として扱い、llama.cppの入力評価と生成は期限到達時に中断する。生成エラーは正常終了として扱わず、圧縮エラーとしてUIまで返して結果欄に保持する
- 起動時に保存済みのモデルと圧縮レベルを読み込み、固定圧縮プロンプトまで準備する
- モデルと圧縮プロンプトをプロセス内で再利用し、必要トークン数に応じてコンテキストを1024、2048、4096の最小段階へ切り替える
- 入力停止から0.45秒後、貼り付け・ドロップ時は0.05秒後に、現在の原文と設定に対応する生成直前のモデル内部状態を1件だけ先読みする。圧縮時はセッションを複製せず、そのまま1回だけ生成処理へ引き渡す。圧縮結果は保存せず、同じ入力の再実行でも毎回LLMの生成処理を行う
- 起動時の固定圧縮プロンプト準備では、最初のセッション複製と1トークン分の生成経路を先に実行し、ユーザー入力へ影響しない一時セッションを破棄する。初回圧縮だけに残るコールドスタートを起動準備へ移し、実際の入力、生成上限、サンプラー、検証条件は変更しない
- LLM出力は必須語、数値、否定、対象限定、状態維持、検証条件などを確認し、不正な場合は原文を返す
- 前処理では明示的な訂正や意味を持たない断片だけを整理し、後処理では「非表示」「状態維持」「触らない」など意味が同じ短い表現を認識する。検証失敗時も原文節を無条件に戻さず、必須要件を満たす短い復元候補を再検証してから採用する
- EXE版は1プロセスだけ起動し、2回目以降は既存ウィンドウを前面へ戻して「すでに起動しています」と表示する
- EXE版の×ボタンはウィンドウだけを隠してモデルの読み込み状態を保持する。初回は「OK」と「今後表示しない」を備えた案内を表示し、完全終了はタスクトレイの「終了」から行う
- EXE版はHTTPサーバーを起動せず、WebView2の内部プロトコルとアプリ内bridgeで通信する
- 開発中のサンプル文章は入力欄へ本文を入れるだけで、モデル、圧縮モード、保存設定、圧縮結果を変更しない。完成版ではサンプル用スクリプトとUIを削除する

UIは既定のデスクトップサイズでページ全体を固定し、入力欄と結果欄だけを必要に応じて内部スクロールさせます。タイトル横には現在のモデル種別と圧縮モードを小枠で表示し、設定画面では「標準」と「高圧縮」をセグメント式で切り替えます。CPU版とスレッド数の手動指定は折りたたみ式の「詳細設定」にまとめ、通常時の設定画面を増やしすぎない構成です。入力欄の見出し、サンプル選択、クリア、圧縮ボタンはデスクトップ版の最小幅でも一列を維持し、それ以外の領域はウィンドウ幅に合わせて再配置します。

### 現在の検証結果

2026-07-22時点の最新ソースとリリースEXEで、次を確認済みです。

- `cargo fmt --all -- --check`: 成功
- `cargo metadata --locked --no-deps`: 成功
- `cargo test --workspace`: 223件成功、Hugging Face通信を伴う1件を除外
- `cargo clippy --workspace --all-targets -- -D warnings`: 成功
- AVX-512版: コンパイルと静的リンクに成功
- compatible、AVX2、AVX-512の3種類について、Rustとllama.cppの命令セットを一致させた厳格リリースビルド: 成功
- パッケージ内の実モデルによる入力先読みと5件の要件充足テスト: 成功
- このPCで実行可能な各CPU版を直接起動し、`internal_llm`、`llama_cpp_embedded`、圧縮結果、必須要件の保持をCPU版ごとに検証

最新の `TrimPrompt.exe` は `../TrimPrompt-exe/` に生成済みです。

### 現在の未確認事項

- 2026-07-13時点のレベル2基準30件は旧評価経路で全件合格している
- 2026-07-16時点の旧組み込みランタイムによる最終パイプライン評価30件は20件合格、10件不合格。平均文字数比は0.82を超えており、圧縮精度側の課題が残る
- llama.cpp `b9982` 採用後の旧版との同一条件速度比較と、レベル2基準30件の全件評価は未実施。固定5入力は文章一致ではなく要件充足で3種類のCPU版を確認済み
- AVX-512版はこのPCで実モデル実行まで確認済み。対応CPUであっても無条件に採用せず、初回診断の実測結果と要件充足結果でAVX2版と比較する
- `cargo-audit` が開発環境に未導入のため、今回のllama.cpp更新後の依存関係監査は未実施
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

- `TrimPrompt.exe`（CPU命令セットを判定する起動ランチャー）
- `application/runtime/cpu/TrimPrompt-avx2.exe`
- `application/runtime/cpu/TrimPrompt-avx512.exe`
- `application/runtime/cpu/TrimPrompt-compatible.exe`
- `application/config/`
- `application/resources/prompts/`
- モデル、キャッシュ、ログ、状態を保存する `application/local/`（初回生成時は空）

推論ランタイムはCPU別の内部EXEへ静的に組み込みます。通常は初回診断で自動選択するため、ユーザーが外部ランタイムを導入したりCPU版を手動で選んだりする必要はありません。必要な場合だけ設定の「詳細設定」から対応済みCPU版とスレッド数を上書きできます。

組み込みランタイムは `llama-cpp-4 0.4.2` と llama.cpp `b9982` (`99f3dc32296f825fec94f202da1e9fede1e78cf9`) です。`native` ビルドには依存せず、Rustとllama.cppの両方へ次の命令セットを明示して別々に構築します。AVX-512非対応の開発PCでも安全にクロスビルドできるよう、CPU別ランタイムだけ明示的な `x86_64-pc-windows-msvc` ターゲットとして構築します。

- compatible: SSE4.2基準、AVX系は無効
- AVX2: AVX、AVX2、FMA、F16C、BMI2
- AVX-512: AVX2構成にAVX-512基本セットを追加。VBMI、VNNI、BF16は無効

各内部EXEは、Rust側のコンパイル命令、アプリ側のfeature、llama.cpp側の命令セットが一致することをビルド時とモデル読込時に自己検査します。起動ランチャーと内部EXEの双方がSSE4.2とBMI2を含む必要命令を判定し、対応する内部EXEだけを実行します。CPU、命令セット、モデル、プロンプト、診断条件、推論実装のいずれかが変わった場合は保存済み診断を再利用せず、CPU最適化を取り直します。UIや説明文だけを更新した場合は推論互換IDが変わらないため、測定済み設定をそのまま利用します。

再現可能なCPU別ビルドを維持するため、調整済みの `llama-cpp-sys-4 0.4.2` と対応するllama.cppソースを `application/vendor/llama-cpp-sys-4/` に固定しています。上流のMITライセンスを同梱し、GPUバックエンドとRPCは有効にしていません。

GGUFモデルは約1.97GBあるためパッケージへ同梱せず、初回セットアップ時にHugging Faceから取得して `application/local/models/` に配置します。

同じ出力先へ `-Clean` を付けて更新しても、`application/local/` と `application/config/user/` は削除しません。取得済みモデル、途中まで取得したモデル、設定、キャッシュ、ログ、WebView2状態を保持したまま、EXEとアプリ管理ファイルだけを更新します。既存モデルを使ったパッケージ内品質テストでも、モデルを上書き・削除しません。品質テストはこのPCで実行可能なcompatible、AVX2、AVX-512版をそれぞれ直接起動し、5つの代表入力の必須要件が圧縮結果に残ることを確認します。

別の場所へ出力したい場合は、次のように指定します。

```powershell
powershell -ExecutionPolicy Bypass -File application/tools/package-desktop-release.ps1 -Clean -OutputPath "C:\path\to\TrimPrompt"
```

## ビルドに必要なもの

実行時は自己完結を目指していますが、開発環境で `llama.cpp` 組み込みバックエンドをビルドするには LLVM とVisual Studio付属のCMake/MSVCが必要です。CMakeはVisual Studio 18または2022の標準インストール先から自動検出します。

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

2026-07-13の実LLM全件評価では、当時の実装でレベル2基準30件がこの条件を満たしました。2026-07-16の旧組み込みランタイムによる再評価では20件が合格し、10件が不合格でした。llama.cpp `b9982` 採用後の全30件評価は未実施であり、現行の圧縮精度はまだ確定していません。圧縮プロンプト、前処理、後処理、検証、fixture、モデル、組み込みランタイムのいずれかを変更した場合は、関連ケースを個別確認した後、必ず全30件を再実行します。

評価の詳細と引き継ぎ情報:

```text
資料/評価メモ/Codex引き継ぎ_レベル2圧縮改善.md
```

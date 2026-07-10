# 内部LLM実行方針

## 目的

LM Studioを常に起動しておかなくても、Prompt Compressor自身がローカルLLMを実行できるようにする。外部サーバー利用は引き続き選べるものとし、内部実行への移行を強制しない。

## 実装方針

- 実行エンジンは `llama.cpp` の `llama-server.exe` を使う。
- アプリは `internal_llm` プロファイルが選ばれた最初の圧縮要求でサーバーを起動する。
- サーバーは `127.0.0.1:8788` に限定して待受し、`/health` が応答するまで圧縮要求を送らない。
- 起動済みのプロセスはアプリの実行中に再利用し、アプリ終了時に終了する。
- 圧縮要求は既存のOpenAI互換 `POST /v1/chat/completions` を使うため、LM Studioと共通の応答解析を利用する。

## 設定の分担

- `application/config/runtime-backends.yaml`: 実行方式、サーバーの場所、ポート、起動待ち時間、CPU/GPU設定
- `application/config/model-catalog.yaml`: GGUFモデルの配置、コンテキスト長、出力上限、テンプレート
- `application/config/compression-profiles.yaml`: UIとCLIで選ぶ `internal_llm` プロファイル
- `application/local/runtimes/`: `llama-server.exe` の配置場所
- `application/local/models/`: GGUFモデルの配置場所

UIにはモデルパスや実行コマンドを常時表示せず、通常はプロファイルを選ぶだけで使えるようにする。詳細設定はYAMLにまとめる。

## 有効化手順

1. `application/local/runtimes/llama-server.exe` を配置する。
2. `application/local/models/qwen3-1.7b-q4_k_m.gguf` を配置する。別モデルを使う場合は `model-catalog.yaml` を変更する。
3. CLIで `--profile internal_llm` を指定する。現在の Web UI ではモデル選択を LM Studio と Sarashina に限定している。

必要ファイルがない場合、圧縮結果は原文を返し、`should_send_original=true` になる。LM Studioプロファイルや既存の一回実行プロファイルには影響しない。

## 将来の拡張

- デスクトップ版でのモデルダウンロード、配置、更新
- 実行中モデルとメモリ使用量の状態表示
- CPUスレッド数とGPUレイヤー数の設定画面
- 休止時の自動停止と次回要求時の再起動
- モデルごとのプロファイル追加

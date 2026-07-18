(() => {
  const sampleSelect = document.querySelector("#sampleSelect");
  const promptInput = document.querySelector("#promptInput");
  if (!sampleSelect || !promptInput) {
    return;
  }

  // 完成版では削除する補助機能のため、設定や圧縮処理には接続しない。
  const samplePrompts = Object.freeze({
    search:
      "React と TypeScript で作っている管理画面の検索一覧を直してほしいです。今の実装では検索欄へ一文字入力するたびに API が呼ばれてしまい、入力を消した時にも通信が走るので、利用者から画面が落ち着かないと言われています。前にも似た修正をお願いした気がしますが、今回は検索ボタンを押した時、または検索欄で Enter を押した時だけ GET /api/customers を呼ぶ形にしてください。入力中は通信しないでください。検索条件は既存の useSearchParams で URL クエリへ保存し、ブラウザの戻る・進むでも復元できる状態を維持してください。新しい検索を実行した時だけ page を 1 に戻し、ページ移動だけの場合は keyword と status を消さないでください。連打で古いレスポンスが後から表示されないよう AbortController も使ってください。既存コンポーネントの分割方法や CSS はなるべく変えず、画面全体の作り直しは避けてください。Vitest ではボタン、Enter、入力中に呼ばれないこと、古いリクエストの中断を確認してください。",
    bugfix:
      "Next.js の注文作成 API について、最近フロント側の入力漏れで 500 が増えているので、ひとまず落ち方を正したいです。対象は POST /api/orders です。customerId が未指定、null、空文字、空白だけのいずれかなら在庫引当へ進む前に HTTP 400 を返し、JSON の error.code は INVALID_CUSTOMER にしてください。requestId がない場合も HTTP 400 と INVALID_REQUEST_ID を返してください。ただし成功時のレスポンス形式、orderId の採番、在庫引当、決済予約、既存の監査ログ形式は変更しないでください。同じ requestId が再送された時は二重注文を作らず、現在の冪等性処理をそのまま通してください。エラー本文に受け取った customerId や個人情報を丸ごと入れないでください。テストは正常系、各不正値、同一 requestId の再送を追加し、既存テストの書き方に合わせてください。今回の目的は入力検証の追加なので、注文処理全体のリファクタリングや DB スキーマ変更までは行わないでください。",
    settings:
      "Windows のデスクトップアプリで、終了するたびに設定が初期化されるのが不便なので保存できるようにしてください。利用者が毎回選び直しているのはモデル、圧縮レベル、ライト・ダークのテーマ、ウィンドウサイズです。この4項目だけを user-settings.json に保存し、次回起動時に復元してください。一方で、入力したプロンプト本文、圧縮結果、クリップボードの内容、最近開いたファイルパスは機密情報を含む可能性があるため保存しないでください。保存は一時ファイルへ書いてから置換する方式にし、書き込み途中でアプリが落ちても設定ファイルが半端な JSON になりにくくしてください。設定ファイルが存在しない、読み取れない、壊れている場合でもアプリの起動は止めず、既定値で続行して警告ログだけ残してください。ログへ設定値そのものや本文は出さないでください。既存設定に未知のキーがあっても削除せず、将来のバージョンと共存できるようにしてください。保存先は現在の application/local/state 配下を維持し、レジストリへの移行は不要です。",
    csv:
      "管理画面の CSV 一括登録が取引先ごとに文字化けしたり途中で止まったりするので、読み込み部分を安定させたいです。アップロードされたファイルが UTF-8、UTF-8 BOM 付き、Shift_JIS のどれかを判定して、いずれも同じ columns マッピングへ渡してください。先頭数行を見ただけで文字コードを決めてデータを欠落させるのは避け、判定できない場合は UNSUPPORTED_ENCODING と対象ファイル名だけを返してください。10MB を超えるファイルは内容を読み込む前に拒否し、INVALID_FILE_SIZE を返してください。空行は無視して構いませんが、値が空の列を勝手に詰めないでください。既存の dryRun、エラー行番号、重複判定、成功件数と失敗件数の集計は維持してください。CSV の全内容をログへ出すことは禁止し、エラー時は行番号と列名までにしてください。大きいファイルでも UI が固まらないようストリームで処理し、途中失敗時に一部だけ DB 登録されないことをテストしてください。今回は画面レイアウトの変更や新しいアップロード画面の追加は不要です。",
    ci:
      "GitHub Actions の Node.js CI が15分以上かかり、同じ依存関係を毎回取得しているようなので改善してください。対象は pull_request と main への push で動く workflow です。Node.js 22 を使う現在の設定、npm ci、npm test、npm run lint、npm run typecheck の順序と実行条件は変更しないでください。actions/setup-node の cache: npm を利用し、package-lock.json をキャッシュキーへ反映してください。モノレポなので packages/*/package-lock.json も存在する場合は dependency-cache の対象に含めますが、node_modules 自体はキャッシュしないでください。キャッシュの復元に失敗した場合でもテストは通常どおり続行し、キャッシュ障害だけで CI 全体を失敗させないでください。ログで cache hit、cache miss、使用した lockfile が分かるようにしてください。外部 fork からの pull_request でも secrets を要求しない構成を維持してください。権限は contents: read を基本とし、不要な write 権限を追加しないでください。変更後はキャッシュなしの初回とキャッシュありの2回目の両方を確認してください。",
    logs:
      "本番で決済確認 API の応答が時々遅くなるので、添付した application.log を調べて原因候補を整理してください。単に遅い行を並べるのではなく、同じ request_id と trace_id をたどり、POST /api/payments/confirm の受付から DB 更新、外部 PSP 呼び出し、レスポンスまでの時間を区間ごとに比較してください。2026-07-08 14:00 から 15:00 の範囲を優先し、latency_ms が3000を超えるリクエストと正常なリクエストを少なくとも3件ずつ比べてください。timeout、retry_count、pool_wait_ms に相関があるかも見てください。断定できない内容は推測と明記し、ログにない事実を補わないでください。カード番号、access_token、email が含まれていた場合は回答へ転載せずマスクしてください。最終結果は、観測事実、可能性の高い原因、可能性の低い原因、追加で必要な計測、すぐできる暫定対応の順でまとめてください。コード変更はまだ行わず、調査結果と再現・確認コマンドだけを提示してください。"
  });

  sampleSelect.addEventListener("change", () => {
    const sample = samplePrompts[sampleSelect.value];
    if (!sample) {
      return;
    }

    promptInput.value = sample;
    promptInput.dispatchEvent(new Event("input", { bubbles: true }));
    sampleSelect.value = "";
    promptInput.focus();
  });
})();

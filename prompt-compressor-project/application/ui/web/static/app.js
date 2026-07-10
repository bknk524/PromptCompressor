const profileSelect = document.querySelector("#profileSelect");
const levelInput = document.querySelector("#levelInput");
const levelValue = document.querySelector("#levelValue");
const levelHelp = document.querySelector("#levelHelp");
const themeToggle = document.querySelector("#themeToggle");
const settingsButton = document.querySelector("#settingsButton");
const settingsMenu = document.querySelector("#settingsMenu");
const windowDragRegion = document.querySelector("#windowDragRegion");
const windowMinimizeButton = document.querySelector("#windowMinimizeButton");
const windowMaximizeButton = document.querySelector("#windowMaximizeButton");
const windowCloseButton = document.querySelector("#windowCloseButton");
const compressButton = document.querySelector("#compressButton");
const clearInputButton = document.querySelector("#clearInputButton");
const sampleSelect = document.querySelector("#sampleSelect");
const copyButton = document.querySelector("#copyButton");
const promptInput = document.querySelector("#promptInput");
const promptOutput = document.querySelector("#promptOutput");
const modelStatus = document.querySelector("#modelStatus");
const workStatus = document.querySelector("#workStatus");
const tokenComparison = document.querySelector("#tokenComparison");
const characterComparison = document.querySelector("#characterComparison");
const tokenRatioValue = document.querySelector("#tokenRatioValue");
const characterRatioValue = document.querySelector("#characterRatioValue");
const fallbackValue = document.querySelector("#fallbackValue");
const settingsSummary = document.querySelector("#settingsSummary");
const compressionLatency = document.querySelector("#compressionLatency");
const settingsStorageKey = "promptCompressorSettingsV3";
const legacySettingsStorageKey = "promptCompressorSettingsV2";
const themeStorageKey = "promptCompressorThemeV1";
const fixedCompressionMode = "codex_optimized";
const fixedTaskType = "coding";
const compressionLevelMin = 1;
const compressionLevelMax = 3;
let isCompressing = false;
let prepareTimerId = 0;
let inFlightPrepareKey = "";
let readyPrepareKey = "";

const compressionLevelDetails = {
  1: {
    name: "控えめ",
    description: "1 控えめ: 表現を残しつつ重複だけ削る"
  },
  2: {
    name: "標準",
    description: "2 標準: 要件を保ちながら短く圧縮"
  },
  3: {
    name: "強め",
    description: "3 強め: 高いほど強く圧縮し、短い同義表現も使う"
  }
};

const samplePrompts = {
  search: {
    level: 2,
    text: [
      "React の検索画面で、検索ボタンを押したときだけ API を呼び出してください。",
      "既存の useSearchParams による URL クエリ管理は維持し、ページ番号変更時も検索状態を保持してください。",
      "TypeScript の既存構造を活かし、大規模なリファクタリングは避けてください。"
    ].join("\n")
  },
  bugfix: {
    level: 2,
    text: [
      "Next.js の POST /api/orders が空の customerId で 500 を返します。",
      "入力検証を追加し、HTTP 400 と既存の INVALID_CUSTOMER エラーコードを返してください。",
      "成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。"
    ].join("\n")
  },
  tests: {
    level: 2,
    text: [
      "TypeScript の parseDateRange 関数に Vitest テストを追加してください。",
      "YYYY-MM-DD の正常値、終了日が開始日より前、無効日付、空文字列を検証してください。",
      "実装コードと既存テストの名前は変更せず、境界値を含めてください。"
    ].join("\n")
  },
  refactor: {
    level: 2,
    text: [
      "React の UserTable を UserTable と UserTableRow に分割し、行の表示責務を分離してください。",
      "公開 props、data-testid、ソートとページネーションの挙動は維持してください。",
      "CSS の見た目と E2E テストは変更せず、不要な再レンダリングを増やさないでください。"
    ].join("\n")
  },
  logs: {
    level: 1,
    text: [
      "本番ログを解析し、注文送信が失敗する原因候補を優先度順に整理してください。",
      "2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service",
      "時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。"
    ].join("\n")
  },
  design: {
    level: 2,
    text: [
      "社内向け申請画面の通知方式を、メール通知とアプリ内通知から選定してください。",
      "管理者は未処理申請を見落とさず、一般利用者には承認結果だけを通知します。",
      "月額コストは 3 万円以下、個人情報を外部サービスへ送信しないことが条件です。",
      "結論、採用理由、未解決事項の順で提案してください。"
    ].join("\n")
  }
};

levelInput.addEventListener("input", () => {
  renderCompressionLevel();
  saveSettings();
  updateSettingsSummary();
  scheduleCompressionPrepare();
});

themeToggle.addEventListener("change", () => {
  const theme = themeToggle.checked ? "dark" : "light";
  applyTheme(theme);
  saveSettings();
});

settingsButton.addEventListener("click", () => {
  toggleSettingsMenu();
});

document.addEventListener("click", (event) => {
  if (
    settingsMenu.hidden ||
    settingsMenu.contains(event.target) ||
    settingsButton.contains(event.target)
  ) {
    return;
  }

  closeSettingsMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSettingsMenu();
    settingsButton.focus();
  }
});

windowDragRegion.addEventListener("mousedown", (event) => {
  if (event.button !== 0) {
    return;
  }

  event.preventDefault();
  postDesktopMessage(event.detail === 2 ? "window:maximize" : "window:drag");
});

windowMinimizeButton.addEventListener("click", () => {
  postDesktopMessage("window:minimize");
});

windowMaximizeButton.addEventListener("click", () => {
  postDesktopMessage("window:maximize");
});

windowCloseButton.addEventListener("click", () => {
  postDesktopMessage("window:close");
});

profileSelect.addEventListener("change", () => {
  saveSettings();
  updateSettingsSummary();
  scheduleCompressionPrepare();
});

sampleSelect.addEventListener("change", () => {
  loadSelectedSample();
});

function loadSelectedSample() {
  const sample = samplePrompts[sampleSelect.value];
  if (!sample) {
    return;
  }

  promptInput.value = sample.text;
  levelInput.value = clampCompressionLevel(sample.level);
  renderCompressionLevel();
  clearResultLists();
  saveSettings();
  updateSettingsSummary();
  scheduleCompressionPrepare();
  promptInput.focus();
}

copyButton.addEventListener("click", async () => {
  if (!promptOutput.value.trim()) {
    return;
  }
  await navigator.clipboard.writeText(promptOutput.value);
  setWorkStatus("コピー済み", "");
});

clearInputButton.addEventListener("click", () => {
  promptInput.value = "";
  sampleSelect.value = "";
  clearResultLists();
  setWorkStatus("入力待ち", "");
  promptInput.focus();
});

compressButton.addEventListener("click", async () => {
  const inputText = promptInput.value.trim();
  if (!inputText) {
    setWorkStatus("入力待ち", "error");
    promptInput.focus();
    return;
  }

  isCompressing = true;
  setLoading(true);
  setWorkStatus("圧縮中", "busy");

  try {
    const response = await fetch("/api/compress", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        input_text: inputText,
        profile: profileSelect.value,
        task_type: fixedTaskType,
        compression_mode: fixedCompressionMode,
        compression_level: Number(levelInput.value),
        constraints: defaultCompressionConstraints()
      })
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Compression failed");
    }

    renderResult(payload);
    const copied = await copyTextToClipboard(payload.distilled_prompt || "");
    const completionMessage = buildCompletionMessage(payload, copied);
    const statusText = copied ? "圧縮完了・コピー済み" : "圧縮完了・コピー失敗";
    setWorkStatus(statusText, copied && !payload.should_send_original ? "" : "error");
    showCompletionNotice(completionMessage);
  } catch (error) {
    setWorkStatus("エラー", "error");
    promptOutput.value = String(error.message || error);
    clearResultLists();
  } finally {
    setLoading(false);
    isCompressing = false;
    refreshRuntimeStatus();
  }
});

async function loadProfiles() {
  const response = await fetch("/api/profiles");
  const payload = await response.json();
  const saved = await loadSettings();
  const profiles = payload.profiles || [];
  const defaultProfile = payload.default_profile || profiles[0]?.id || "";
  const preferredProfile = profiles.some((profile) => profile.id === saved.profile)
    ? saved.profile
    : defaultProfile;
  profileSelect.textContent = "";
  for (const profile of profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.label;
    if (profile.id === preferredProfile) {
      option.selected = true;
    }
    profileSelect.append(option);
  }
  if (saved.level !== undefined) {
    levelInput.value = clampCompressionLevel(saved.level);
    renderCompressionLevel();
  }
  if (saved.theme) {
    applyTheme(saved.theme);
  }
  updateSettingsSummary();
  saveSettings();
  scheduleCompressionPrepare(50);
}

function scheduleCompressionPrepare(delayMs = 350) {
  window.clearTimeout(prepareTimerId);
  prepareTimerId = window.setTimeout(() => {
    prepareCompressionSelection();
  }, delayMs);
}

async function prepareCompressionSelection() {
  const payload = currentPreparePayload();
  if (!payload) {
    return;
  }

  const prepareKey = JSON.stringify(payload);
  if (prepareKey === readyPrepareKey || prepareKey === inFlightPrepareKey) {
    return;
  }

  inFlightPrepareKey = prepareKey;
  setModelStatus("モデル準備中", "busy");

  try {
    const response = await fetch("/api/prepare-compression", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "prepare failed");
    }

    if (prepareKey !== JSON.stringify(currentPreparePayload())) {
      return;
    }

    if (result.prepared) {
      readyPrepareKey = prepareKey;
      setModelStatus(result.message || "モデル準備完了", "");
    } else {
      setModelStatus(result.message || "待機中", "");
    }
  } catch (_error) {
    setModelStatus("準備エラー", "error");
  } finally {
    if (inFlightPrepareKey === prepareKey) {
      inFlightPrepareKey = "";
    }
  }
}

function currentPreparePayload() {
  if (!profileSelect.value) {
    return null;
  }

  return {
    profile: profileSelect.value,
    task_type: fixedTaskType,
    compression_mode: fixedCompressionMode,
    compression_level: Number(clampCompressionLevel(levelInput.value)),
    constraints: defaultCompressionConstraints()
  };
}

function defaultCompressionConstraints() {
  return {
    preserve_code_blocks: true,
    preserve_file_names: true,
    preserve_error_messages: true,
    preserve_numbers: true,
    preserve_negations: true
  };
}

function renderResult(result) {
  promptOutput.value = result.distilled_prompt || "";
  const metrics = result.metrics || {};
  tokenComparison.textContent = formatComparison(
    metrics.input_tokens_est,
    metrics.output_tokens_est
  );
  tokenRatioValue.textContent = Number.isFinite(metrics.compression_ratio)
    ? `${Math.round(metrics.compression_ratio * 100)}%`
    : "-";
  characterComparison.textContent = formatComparison(
    metrics.input_characters,
    metrics.output_characters
  );
  characterRatioValue.textContent = Number.isFinite(metrics.character_ratio)
    ? `${Math.round(metrics.character_ratio * 100)}%`
    : "-";
  fallbackValue.textContent = result.should_send_original ? "はい" : "いいえ";
  compressionLatency.textContent = formatLatencySeconds(metrics.latency_ms);
}

function clearResultLists() {
  promptOutput.value = "";
  tokenComparison.textContent = "-";
  tokenRatioValue.textContent = "-";
  characterComparison.textContent = "-";
  characterRatioValue.textContent = "-";
  fallbackValue.textContent = "-";
  compressionLatency.textContent = "";
}

function formatComparison(input, output) {
  if (!Number.isFinite(input) || !Number.isFinite(output)) {
    return "-";
  }
  return `${input} → ${output}`;
}

function formatLatencySeconds(latencyMs) {
  if (!Number.isFinite(latencyMs)) {
    return "";
  }

  const seconds = latencyMs / 1000;
  return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`;
}

function setLoading(isLoading) {
  compressButton.disabled = isLoading;
  clearInputButton.disabled = isLoading;
  compressButton.textContent = isLoading ? "圧縮中" : "圧縮する";
}

function setModelStatus(text, state) {
  updateStatusPill(modelStatus, "モデル", text, state);
}

function setWorkStatus(text, state) {
  updateStatusPill(workStatus, "作業", text, state);
}

function updateStatusPill(element, label, text, state) {
  element.textContent = `${label}: ${text}`;
  element.className = `status-pill ${state || ""}`.trim();
}

async function copyTextToClipboard(text) {
  if (!text.trim()) {
    return false;
  }

  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_error) {
    if (await copyTextWithNativeApi(text)) {
      return true;
    }

    promptOutput.focus();
    promptOutput.select();
    try {
      return document.execCommand("copy");
    } catch (_fallbackError) {
      return false;
    } finally {
      promptOutput.setSelectionRange(0, 0);
    }
  }
}

async function copyTextWithNativeApi(text) {
  try {
    const response = await fetch("/api/clipboard", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });
    if (!response.ok) {
      return false;
    }

    const payload = await response.json();
    return Boolean(payload.copied);
  } catch (_error) {
    return false;
  }
}

function buildCompletionMessage(result, copied) {
  const output = summarizeText(result.distilled_prompt || "");
  const metrics = result.metrics || {};
  const tokenText = formatComparison(metrics.input_tokens_est, metrics.output_tokens_est);
  const characterText = formatComparison(metrics.input_characters, metrics.output_characters);
  const copyText = copied ? "クリップボードにコピーしました" : "クリップボードへのコピーに失敗しました";
  return `圧縮完了。${copyText}。圧縮後: ${output}（トークン ${tokenText} / 文字数 ${characterText}）`;
}

function summarizeText(text) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (compact.length <= 120) {
    return compact || "-";
  }
  return `${compact.slice(0, 117)}...`;
}

async function notifyCompressionComplete(message) {
  await notifyWindowsCompletion(message);
}

async function notifyWindowsCompletion(message) {
  if (postDesktopNotification("圧縮完了", message)) {
    return true;
  }

  try {
    const response = await fetch("/api/windows-notification", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        title: "圧縮完了",
        body: message
      })
    });
    if (!response.ok) {
      return false;
    }

    const payload = await response.json();
    return Boolean(payload.notified);
  } catch (_error) {
    return false;
  }
}

function showCompletionNotice(message) {
  notifyCompressionComplete(message);
}

async function refreshRuntimeStatus() {
  try {
    const response = await fetch("/api/runtime-status");
    if (!response.ok) {
      return;
    }
    const status = await response.json();
    switch (status.phase) {
      case "loading":
        setModelStatus(status.message || "モデル読み込み中", "busy");
        break;
      case "ready":
        setModelStatus(status.message || "モデル準備完了", "");
        break;
      case "error":
        setModelStatus(status.message || "モデル読み込み失敗", "error");
        break;
      case "skipped":
        setModelStatus(status.message || "待機中", "");
        break;
      default:
        setModelStatus(status.message || "待機中", "");
        break;
    }
  } catch (_error) {
    // The UI can keep working even if the status probe races server startup.
  }
}

function updateSettingsSummary() {
  const profileLabel =
    profileSelect.selectedOptions[0]?.textContent || "Standard";
  const levelDetail = compressionLevelDetails[levelInput.value];
  const levelLabel = levelDetail ? `${levelInput.value} ${levelDetail.name}` : levelInput.value;
  settingsSummary.textContent = `${profileLabel} / Level ${levelLabel}`;
}

function clampCompressionLevel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "2";
  }

  return String(
    Math.min(compressionLevelMax, Math.max(compressionLevelMin, Math.round(numeric)))
  );
}

function renderCompressionLevel() {
  levelInput.value = clampCompressionLevel(levelInput.value);
  levelValue.textContent = levelInput.value;
  levelHelp.textContent =
    compressionLevelDetails[levelInput.value]?.description ||
    compressionLevelDetails[2].description;
}

function initializeTheme() {
  const savedTheme = loadLocalSettings().theme || localStorage.getItem(themeStorageKey);
  applyTheme(savedTheme === "dark" ? "dark" : "light");
}

function applyTheme(theme) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  document.body.dataset.theme = isDark ? "dark" : "light";
  themeToggle.checked = isDark;
  notifyDesktopTheme(isDark ? "dark" : "light");
}

function notifyDesktopTheme(theme) {
  postDesktopMessage(`theme:${theme}`);
}

function postDesktopNotification(title, body) {
  return postDesktopMessage(
    `notification:${JSON.stringify({
      title,
      body
    })}`
  );
}

function postDesktopMessage(message) {
  if (!window.ipc || typeof window.ipc.postMessage !== "function") {
    return false;
  }

  window.ipc.postMessage(message);
  return true;
}

function toggleSettingsMenu() {
  if (settingsMenu.hidden) {
    openSettingsMenu();
  } else {
    closeSettingsMenu();
  }
}

function openSettingsMenu() {
  settingsMenu.hidden = false;
  settingsButton.setAttribute("aria-expanded", "true");
}

function closeSettingsMenu() {
  settingsMenu.hidden = true;
  settingsButton.setAttribute("aria-expanded", "false");
}

window.promptCompressorOpenSettings = () => {
  window.scrollTo({ top: 0, behavior: "auto" });
  openSettingsMenu();
  settingsButton.focus();
};

async function loadSettings() {
  const localSettings = loadLocalSettings();
  const serverSettings = await loadServerSettings();
  return { ...localSettings, ...serverSettings };
}

function loadLocalSettings() {
  try {
    const current = JSON.parse(localStorage.getItem(settingsStorageKey) || "{}");
    const theme = localStorage.getItem(themeStorageKey);
    if (Object.keys(current).length > 0) {
      return normalizeLoadedSettings({ ...current, theme: current.theme || theme });
    }

    const legacy = JSON.parse(localStorage.getItem(legacySettingsStorageKey) || "{}");
    const migrated = legacy.profile === "lmstudio" ? { ...legacy, profile: undefined } : legacy;
    return normalizeLoadedSettings({ ...migrated, theme });
  } catch (_error) {
    return {};
  }
}

async function loadServerSettings() {
  try {
    const response = await fetch("/api/settings");
    if (!response.ok) {
      return {};
    }
    return normalizeLoadedSettings(await response.json());
  } catch (_error) {
    return {};
  }
}

function normalizeLoadedSettings(settings) {
  const normalized = {};
  for (const key of ["profile", "level", "theme"]) {
    const value = settings[key];
    if (value !== undefined && value !== null && value !== "") {
      normalized[key] = value;
    }
  }
  return normalized;
}

function saveSettings() {
  const settings = collectSettings();
  localStorage.setItem(
    settingsStorageKey,
    JSON.stringify(settings)
  );
  localStorage.setItem(themeStorageKey, settings.theme);
  saveSettingsToServer(settings);
}

function collectSettings() {
  return {
    schema_version: 1,
    profile: profileSelect.value,
    level: Number(clampCompressionLevel(levelInput.value)),
    theme: themeToggle.checked ? "dark" : "light"
  };
}

async function saveSettingsToServer(settings) {
  try {
    await fetch("/api/settings", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(settings)
    });
  } catch (_error) {
    // localStorage remains as a fallback when the local API is unavailable.
  }
}

initializeTheme();
renderCompressionLevel();

loadProfiles()
  .then(() => {
    clearResultLists();
    refreshRuntimeStatus();
    setInterval(refreshRuntimeStatus, 1500);
  })
  .catch((error) => {
    setModelStatus("設定読み込み失敗", "error");
    setWorkStatus("エラー", "error");
    promptOutput.value = String(error.message || error);
  });

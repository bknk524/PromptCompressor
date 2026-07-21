const profileSelect = document.querySelector("#profileSelect");
const levelInput = document.querySelector("#levelInput");
const levelButtons = [...document.querySelectorAll("[data-compression-level]")];
const levelHelp = document.querySelector("#levelHelp");
const themeToggle = document.querySelector("#themeToggle");
const settingsButton = document.querySelector("#settingsButton");
const settingsMenu = document.querySelector("#settingsMenu");
const cpuEngineSelect = document.querySelector("#cpuEngineSelect");
const threadModeSelect = document.querySelector("#threadModeSelect");
const manualThreadInputs = document.querySelector("#manualThreadInputs");
const generationThreadsInput = document.querySelector("#generationThreadsInput");
const batchThreadsInput = document.querySelector("#batchThreadsInput");
const runtimeCurrentValue = document.querySelector("#runtimeCurrentValue");
const runtimeApplyButton = document.querySelector("#runtimeApplyButton");
const runtimeTuningResetButton = document.querySelector("#runtimeTuningResetButton");
const windowDragRegion = document.querySelector("#windowDragRegion");
const windowMinimizeButton = document.querySelector("#windowMinimizeButton");
const windowMaximizeButton = document.querySelector("#windowMaximizeButton");
const windowCloseButton = document.querySelector("#windowCloseButton");
const compressButton = document.querySelector("#compressButton");
const clearInputButton = document.querySelector("#clearInputButton");
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
const resultNotice = document.querySelector("#resultNotice");
const resultNoticeDetail = document.querySelector("#resultNoticeDetail");
const modelSetup = document.querySelector("#modelSetup");
const modelSetupTitle = document.querySelector("#modelSetupTitle");
const modelSetupDetail = document.querySelector("#modelSetupDetail");
const modelDownloadProgress = document.querySelector("#modelDownloadProgress");
const modelInstallButton = document.querySelector("#modelInstallButton");
const modelCancelButton = document.querySelector("#modelCancelButton");
const settingsSummary = document.querySelector("#settingsSummary");
const compressionLatency = document.querySelector("#compressionLatency");
const runtimeSetupScreen = document.querySelector("#runtimeSetupScreen");
const runtimeSetupTitle = document.querySelector("#runtimeSetupTitle");
const runtimeSetupDetail = document.querySelector("#runtimeSetupDetail");
const settingsStorageKey = "trimPromptSettingsV1";
const legacySettingsStorageKeys = [
  "promptCompressorSettingsV3",
  "promptCompressorSettingsV2"
];
const themeStorageKey = "trimPromptThemeV1";
const legacyThemeStorageKey = "promptCompressorThemeV1";
const compressionLevelMin = 2;
const compressionLevelMax = 3;
const settingsSaveDelayMs = 250;
const inputPrepareDelayMs = 450;
const pastedInputPrepareDelayMs = 50;
let isCompressing = false;
let prepareTimerId = 0;
let inputPrepareTimerId = 0;
let settingsSaveTimerId = 0;
let settingsSaveInFlight = false;
let pendingServerSettings = null;
let lastLocalSettingsJson = "";
let lastServerSettingsFingerprint = "";
let inFlightPrepareKey = "";
let readyPrepareKey = "";
let inFlightInputPrepareKey = "";
let readyInputPrepareKey = "";
let inputPreparePromise = null;
let runtimeSetupInFlight = false;
let modelInstalled = false;
let modelRuntimeReady = false;
let modelInstallInFlight = false;
let modelCancelRequested = false;
let activeModelDownloadProfile = "";
let selectedModelStatus = null;
let runtimeConfiguration = null;
let runtimePreferencesBaseline = "";

const compressionLevelDetails = {
  2: {
    name: "標準",
    description: "要件を保ちながらバランスよく圧縮"
  },
  3: {
    name: "高圧縮",
    description: "短い表現を使い、さらに圧縮率を高める"
  }
};

const cpuEngineLabels = {
  compatible: "SSE4.2 互換",
  avx2: "AVX2",
  avx512: "AVX-512"
};

for (const levelButton of levelButtons) {
  levelButton.addEventListener("click", () => {
    const nextLevel = clampCompressionLevel(levelButton.dataset.compressionLevel);
    if (nextLevel === levelInput.value) {
      return;
    }
    levelInput.value = nextLevel;
    renderCompressionLevel();
    saveSettings();
    updateSettingsSummary();
    invalidatePreparedInput();
    scheduleCompressionPrepare();
  });
}

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
  modelRuntimeReady = false;
  updateCompressButtonState();
  saveSettings();
  updateSettingsSummary();
  readyPrepareKey = "";
  invalidatePreparedInput();
  refreshModelAvailability();
  refreshRuntimeConfiguration();
});

runtimeTuningResetButton.addEventListener("click", resetRuntimeTuning);
runtimeApplyButton.addEventListener("click", applyRuntimeSettings);

for (const control of [
  cpuEngineSelect,
  threadModeSelect,
  generationThreadsInput,
  batchThreadsInput
]) {
  control.addEventListener("change", () => {
    updateRuntimeControls();
    saveSettings();
  });
}

promptInput.addEventListener("input", (event) => {
  invalidatePreparedInput();
  const pasted = event.inputType === "insertFromPaste" || event.inputType === "insertFromDrop";
  scheduleInputPrepare(pasted ? pastedInputPrepareDelayMs : inputPrepareDelayMs);
});

copyButton.addEventListener("click", async () => {
  if (!promptOutput.value.trim()) {
    return;
  }
  const copied = await copyTextToClipboard(promptOutput.value);
  if (!copied) {
    setWorkStatus("コピー失敗", "error");
  }
});

clearInputButton.addEventListener("click", () => {
  promptInput.value = "";
  invalidatePreparedInput();
  clearResultLists();
  setWorkStatus("入力待ち", "");
  promptInput.focus();
});

compressButton.addEventListener("click", async () => {
  const compressionStartedAt = performance.now();
  if (!modelInstalled) {
    setWorkStatus("モデル取得待ち", "error");
    modelInstallButton.focus();
    return;
  }
  if (!modelRuntimeReady) {
    setWorkStatus("モデル準備待ち", "busy");
    return;
  }
  const inputText = promptInput.value.trim();
  if (!inputText) {
    setWorkStatus("入力待ち", "error");
    promptInput.focus();
    return;
  }
  const compressionPayload = {
    input_text: inputText,
    profile: profileSelect.value,
    compression_level: Number(levelInput.value),
    constraints: defaultCompressionConstraints()
  };
  const inputPrepareKey = JSON.stringify(compressionPayload);

  isCompressing = true;
  setLoading(true);
  setWorkStatus("圧縮中", "running");
  window.clearTimeout(inputPrepareTimerId);
  inputPrepareTimerId = 0;

  try {
    if (inputPreparePromise && inFlightInputPrepareKey === inputPrepareKey) {
      await inputPreparePromise;
    }
    const response = await fetch("/api/compress", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(compressionPayload)
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Compression failed");
    }

    renderResult(payload, compressionStartedAt);
    if (payload.should_send_original) {
      setWorkStatus("圧縮せず原文を保持", "error");
      showCompletionNotice(buildFallbackMessage(payload), "原文を保持");
    } else {
      const copied = await copyTextToClipboard(payload.distilled_prompt || "");
      const completionMessage = buildCompletionMessage(payload, copied);
      setWorkStatus("圧縮完了", "");
      showCompletionNotice(completionMessage, "圧縮完了");
    }
  } catch (error) {
    setWorkStatus("エラー", "error");
    clearResultLists();
    promptOutput.value = String(error.message || error);
  } finally {
    setLoading(false);
    isCompressing = false;
    readyInputPrepareKey = "";
    scheduleInputPrepare();
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
  applyRuntimePreferences(saved);
  await refreshRuntimeConfiguration();
  runtimePreferencesBaseline = runtimePreferencesFingerprint();
  updateRuntimeControls();
  updateSettingsSummary();
  saveSettings();
  await refreshModelAvailability(false);
}

async function refreshModelAvailability(prepareWhenInstalled = true) {
  if (!profileSelect.value || modelInstallInFlight) {
    return;
  }
  modelInstalled = false;
  modelRuntimeReady = false;
  updateCompressButtonState();
  try {
    const response = await fetch("/api/model-status", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ profile: profileSelect.value })
    });
    const status = await response.json();
    if (!response.ok) {
      throw new Error(status.error || "model status failed");
    }
    selectedModelStatus = status;
    modelInstalled = !status.requires_install || Boolean(status.installed);
    renderModelSetup(status);
    updateCompressButtonState();
    if (modelInstalled && prepareWhenInstalled) {
      const restarting = await runRuntimeSetupGate();
      if (!restarting) {
        scheduleCompressionPrepare(50);
      }
    }
  } catch (_error) {
    selectedModelStatus = null;
    modelSetup.hidden = false;
    modelSetupTitle.textContent = "モデル状態を確認できませんでした";
    modelSetupDetail.textContent = "設定を確認して再試行してください。";
    modelInstallButton.disabled = true;
    setModelStatus("状態確認エラー", "error");
  }
}

function renderModelSetup(status) {
  if (!status.requires_install || status.installed) {
    modelSetup.hidden = true;
    modelDownloadProgress.hidden = true;
    modelInstallButton.disabled = false;
    modelCancelButton.hidden = true;
    return;
  }

  modelSetup.hidden = false;
  modelSetupTitle.textContent = `${status.label || "ローカルモデル"} の取得が必要です`;
  const source = status.repository ? `Hugging Face: ${status.repository}` : "Hugging Face";
  const size = formatBytes(status.size_bytes);
  const available = Number.isFinite(status.available_bytes)
    ? ` / 空き ${formatBytes(status.available_bytes)}`
    : "";
  const partial = Number(status.partial_downloaded_bytes || 0);
  const resume = partial > 0 ? ` / ${formatBytes(partial)}から再開` : "";
  modelSetupDetail.textContent = `${source} / ${size}${available}${resume}`;
  modelDownloadProgress.hidden = true;
  modelInstallButton.disabled = false;
  modelInstallButton.textContent = "モデルを取得";
  modelCancelButton.hidden = true;
  setModelStatus("モデル未取得", "error");
}

async function installSelectedModel() {
  if (!profileSelect.value || modelInstallInFlight) {
    return;
  }
  const profile = profileSelect.value;
  modelInstallInFlight = true;
  modelCancelRequested = false;
  activeModelDownloadProfile = profile;
  modelInstalled = false;
  modelInstallButton.disabled = true;
  modelInstallButton.textContent = "取得中";
  modelCancelButton.hidden = false;
  modelCancelButton.disabled = false;
  modelDownloadProgress.hidden = false;
  modelDownloadProgress.value = 0;
  setModelStatus("モデル取得中", "busy");
  updateCompressButtonState();

  try {
    const response = await fetch("/api/model-install", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ profile })
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "model install failed");
    }
    if (result.cancelled) {
      modelSetupTitle.textContent = "モデル取得を中止しました";
      setModelStatus("モデル取得中止", "");
    } else {
      setModelReadyStatus();
    }
  } catch (error) {
    modelSetup.hidden = false;
    modelSetupTitle.textContent = modelCancelRequested
      ? "モデル取得を中止しました"
      : "モデル取得に失敗しました";
    modelSetupDetail.textContent = String(error.message || error);
    setModelStatus("モデル取得失敗", "error");
  } finally {
    modelInstallInFlight = false;
    modelCancelRequested = false;
    activeModelDownloadProfile = "";
    modelInstallButton.disabled = false;
    modelInstallButton.textContent = "再試行";
    modelCancelButton.hidden = true;
    modelCancelButton.disabled = false;
    await refreshModelAvailability(false);
    if (modelInstalled) {
      const restarting = await runRuntimeSetupGate();
      if (!restarting) {
        scheduleCompressionPrepare(50);
      }
    }
  }
}

async function cancelSelectedModelDownload() {
  if (!activeModelDownloadProfile || !modelInstallInFlight || modelCancelRequested) {
    return;
  }
  modelCancelRequested = true;
  modelCancelButton.disabled = true;
  setModelStatus("モデル取得を中止中", "busy");
  try {
    const response = await fetch("/api/model-cancel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ profile: activeModelDownloadProfile })
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "model cancel failed");
    }
    modelSetupDetail.textContent = result.message || "モデル取得を中止しています。";
  } catch (error) {
    modelCancelRequested = false;
    modelCancelButton.disabled = false;
    modelSetupDetail.textContent = String(error.message || error);
    setModelStatus("中止要求エラー", "error");
  }
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value < 0) {
    return "容量不明";
  }
  const gibibytes = value / (1024 ** 3);
  if (gibibytes >= 1) {
    return `${gibibytes.toFixed(2)} GiB`;
  }
  return `${Math.max(1, Math.round(value / (1024 ** 2)))} MiB`;
}

modelInstallButton.addEventListener("click", installSelectedModel);
modelCancelButton.addEventListener("click", cancelSelectedModelDownload);

function scheduleCompressionPrepare(delayMs = 350) {
  if (!modelInstalled || modelInstallInFlight) {
    return;
  }
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
  modelRuntimeReady = false;
  updateCompressButtonState();
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
    }
    modelRuntimeReady = true;
    setModelReadyStatus();
    scheduleInputPrepare();
  } catch (_error) {
    modelRuntimeReady = false;
    setModelStatus("準備エラー", "error");
  } finally {
    if (inFlightPrepareKey === prepareKey) {
      inFlightPrepareKey = "";
    }
    updateCompressButtonState();
  }
}

function currentPreparePayload() {
  if (!profileSelect.value) {
    return null;
  }

  return {
    profile: profileSelect.value,
    compression_level: Number(clampCompressionLevel(levelInput.value)),
    constraints: defaultCompressionConstraints()
  };
}

function invalidatePreparedInput() {
  window.clearTimeout(inputPrepareTimerId);
  inputPrepareTimerId = 0;
  readyInputPrepareKey = "";
}

function scheduleInputPrepare(delayMs = inputPrepareDelayMs) {
  const payload = currentInputPreparePayload();
  if (
    !payload ||
    !modelInstalled ||
    !modelRuntimeReady ||
    modelInstallInFlight ||
    isCompressing ||
    inputPrepareTimerId
  ) {
    return;
  }

  const prepareKey = JSON.stringify(payload);
  if (prepareKey === readyInputPrepareKey || prepareKey === inFlightInputPrepareKey) {
    return;
  }
  inputPrepareTimerId = window.setTimeout(() => {
    inputPrepareTimerId = 0;
    prepareInputSelection();
  }, delayMs);
}

async function prepareInputSelection() {
  const payload = currentInputPreparePayload();
  if (!payload || isCompressing) {
    return;
  }
  const prepareKey = JSON.stringify(payload);
  if (prepareKey === readyInputPrepareKey || prepareKey === inFlightInputPrepareKey) {
    return;
  }

  inFlightInputPrepareKey = prepareKey;
  let shouldRetry = false;
  const operation = (async () => {
    const response = await fetch("/api/prepare-input", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (response.status === 429) {
      shouldRetry = true;
      return;
    }
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "input prepare failed");
    }
    if (result.prepared && prepareKey === JSON.stringify(currentInputPreparePayload())) {
      readyInputPrepareKey = prepareKey;
    }
  })();
  inputPreparePromise = operation;

  try {
    await operation;
  } catch (_error) {
    // Input preparation is opportunistic; normal compression remains available.
  } finally {
    if (inFlightInputPrepareKey === prepareKey) {
      inFlightInputPrepareKey = "";
    }
    if (inputPreparePromise === operation) {
      inputPreparePromise = null;
    }
    if (shouldRetry || prepareKey !== JSON.stringify(currentInputPreparePayload())) {
      scheduleInputPrepare(150);
    }
  }
}

function currentInputPreparePayload() {
  const inputText = promptInput.value.trim();
  if (!inputText || !profileSelect.value) {
    return null;
  }
  return {
    input_text: inputText,
    profile: profileSelect.value,
    compression_level: Number(clampCompressionLevel(levelInput.value)),
    constraints: defaultCompressionConstraints()
  };
}

async function runRuntimeSetupGate() {
  if (runtimeSetupInFlight || !modelInstalled || !profileSelect.value) {
    return false;
  }
  const payload = currentPreparePayload();
  if (!payload) {
    revealApplication();
    return false;
  }

  runtimeSetupInFlight = true;
  try {
    const status = await fetchRuntimeSetupStatus(payload);
    if (!status.required) {
      revealApplication();
      return false;
    }

    showRuntimeSetup(
      "初期設定中",
      "このPCに最適な設定を診断しています。完了まで少し時間がかかります"
    );
    const result = await requestRuntimeTuning(payload);
    if (result.restart_required) {
      showRuntimeSetup("初期設定を反映しています", "まもなく自動で再起動します");
      await delay(300);
      if (!postDesktopMessage("app:restart")) {
        window.location.reload();
      }
      return true;
    }

    const updatedStatus = await fetchRuntimeSetupStatus(payload);
    if (updatedStatus.required) {
      throw new Error("runtime setup did not produce a valid saved configuration");
    }

    revealApplication();
    return false;
  } catch (_error) {
    showRuntimeSetupError(
      "初期設定を完了できませんでした",
      "TrimPromptを再起動して、もう一度お試しください"
    );
    return true;
  } finally {
    runtimeSetupInFlight = false;
  }
}

async function fetchRuntimeSetupStatus(payload) {
  const response = await fetch("/api/runtime-setup-status", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const status = await response.json();
  if (!response.ok) {
    throw new Error(status.error || "runtime setup status failed");
  }
  return status;
}

async function requestRuntimeTuning(payload) {
  while (true) {
    const response = await fetch("/api/tune-runtime", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (response.status === 429) {
      await delay(400);
      continue;
    }
    const result = await response.json();
    if (!response.ok || !result.completed) {
      throw new Error(result.error || "runtime tuning failed");
    }
    return result;
  }
}

function showRuntimeSetup(title, detail) {
  document.body.classList.remove("startup-gated");
  document.body.classList.add("runtime-setup-active");
  runtimeSetupScreen.classList.remove("error");
  runtimeSetupScreen.hidden = false;
  runtimeSetupTitle.textContent = title;
  runtimeSetupDetail.textContent = detail;
}

function showRuntimeSetupError(title, detail) {
  showRuntimeSetup(title, detail);
  runtimeSetupScreen.classList.add("error");
}

function revealApplication() {
  document.body.classList.remove("startup-gated", "runtime-setup-active");
  runtimeSetupScreen.classList.remove("error");
  runtimeSetupScreen.hidden = true;
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function applyRuntimePreferences(settings) {
  cpuEngineSelect.value = ["compatible", "avx2", "avx512"].includes(settings.cpu_engine)
    ? settings.cpu_engine
    : "auto";
  threadModeSelect.value = settings.thread_mode === "manual" ? "manual" : "auto";
  generationThreadsInput.value = positiveIntegerOrEmpty(settings.generation_threads);
  batchThreadsInput.value = positiveIntegerOrEmpty(settings.batch_threads);
}

async function refreshRuntimeConfiguration() {
  if (!profileSelect.value) {
    return;
  }
  const profile = profileSelect.value;
  try {
    const response = await fetch("/api/runtime-configuration", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ profile })
    });
    const configuration = await response.json();
    if (!response.ok) {
      throw new Error(configuration.error || "runtime configuration failed");
    }
    if (profileSelect.value !== profile) {
      return;
    }
    runtimeConfiguration = configuration;
    const maximum = Math.max(1, Number(configuration.available_threads) || 1);
    generationThreadsInput.max = String(maximum);
    batchThreadsInput.max = String(maximum);

    for (const option of cpuEngineSelect.options) {
      const availability = configuration.cpu_engines?.find((engine) => engine.id === option.value);
      option.disabled = availability ? !availability.supported : option.value !== "auto";
    }
    if (cpuEngineSelect.selectedOptions[0]?.disabled) {
      cpuEngineSelect.value = "auto";
    }

    if (!positiveIntegerOrEmpty(generationThreadsInput.value)) {
      generationThreadsInput.value = String(configuration.generation_threads || 1);
    }
    if (!positiveIntegerOrEmpty(batchThreadsInput.value)) {
      batchThreadsInput.value = String(configuration.batch_threads || 1);
    }
    renderCurrentRuntimeConfiguration();
    updateRuntimeControls();
  } catch (_error) {
    runtimeConfiguration = null;
    runtimeCurrentValue.textContent = "取得できませんでした";
    updateRuntimeControls();
  }
}

function renderCurrentRuntimeConfiguration() {
  if (!runtimeConfiguration) {
    runtimeCurrentValue.textContent = "取得できませんでした";
    return;
  }
  const engine = cpuEngineLabels[runtimeConfiguration.current_cpu_engine] || "互換版";
  const cpuMode = runtimeConfiguration.current_cpu_mode === "manual" ? "手動" : "自動";
  const threadMode = runtimeConfiguration.current_thread_mode === "manual" ? "手動" : "自動";
  runtimeCurrentValue.textContent =
    `${engine}（${cpuMode}） / 生成 ${runtimeConfiguration.generation_threads}` +
    ` / 入力評価 ${runtimeConfiguration.batch_threads}（${threadMode}）` +
    ` / 入力単位 ${runtimeConfiguration.logical_batch_size}` +
    `→${runtimeConfiguration.physical_batch_size}`;
}

function positiveIntegerOrEmpty(value) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric > 0 ? String(numeric) : "";
}

function runtimePreferencesFingerprint(settings = collectSettings()) {
  return JSON.stringify({
    cpu_engine: settings.cpu_engine,
    thread_mode: settings.thread_mode,
    generation_threads: settings.generation_threads,
    batch_threads: settings.batch_threads
  });
}

function runtimePreferencesAreValid() {
  if (cpuEngineSelect.selectedOptions[0]?.disabled) {
    return false;
  }
  if (threadModeSelect.value !== "manual") {
    return true;
  }
  const maximum = Math.max(1, Number(runtimeConfiguration?.available_threads) || 1);
  return [generationThreadsInput, batchThreadsInput].every((input) => {
    const value = Number(input.value);
    return Number.isInteger(value) && value >= 1 && value <= maximum;
  });
}

function updateRuntimeControls() {
  const manual = threadModeSelect.value === "manual";
  manualThreadInputs.hidden = !manual;
  generationThreadsInput.disabled = !manual;
  batchThreadsInput.disabled = !manual;
  const automatic = cpuEngineSelect.value === "auto" && !manual;
  runtimeTuningResetButton.disabled = !automatic;
  runtimeApplyButton.disabled =
    !runtimePreferencesAreValid() ||
    runtimePreferencesFingerprint() === runtimePreferencesBaseline;
}

async function applyRuntimeSettings() {
  if (runtimeApplyButton.disabled || !runtimePreferencesAreValid()) {
    return;
  }

  const settings = collectSettings();
  saveSettings();
  runtimeApplyButton.disabled = true;
  runtimeApplyButton.textContent = "保存中";
  if (settingsSaveTimerId) {
    clearTimeout(settingsSaveTimerId);
    settingsSaveTimerId = 0;
  }
  pendingServerSettings = null;
  while (settingsSaveInFlight) {
    await delay(25);
  }

  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(settings)
    });
    const saved = await response.json();
    if (!response.ok) {
      throw new Error(saved.error || "runtime settings save failed");
    }
    lastServerSettingsFingerprint = settingsFingerprint(saved);
    closeSettingsMenu();
    setWorkStatus("設定を反映中", "busy");
    if (!postDesktopMessage("app:restart")) {
      window.location.reload();
    }
  } catch (_error) {
    setWorkStatus("設定エラー", "error");
    runtimeApplyButton.textContent = "再起動して適用";
    updateRuntimeControls();
  }
}

async function resetRuntimeTuning() {
  const payload = currentPreparePayload();
  if (!payload || runtimeTuningResetButton.disabled) {
    return;
  }

  runtimeTuningResetButton.disabled = true;
  runtimeTuningResetButton.textContent = "処理中";
  try {
    const response = await fetch("/api/tune-runtime-reset", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok || !result.reset) {
      throw new Error(result.error || "runtime tuning reset failed");
    }

    closeSettingsMenu();
    const restarting = await runRuntimeSetupGate();
    if (!restarting) {
      setWorkStatus("CPU最適化完了", "");
    }
  } catch (_error) {
    setWorkStatus("再調整エラー", "error");
  } finally {
    runtimeTuningResetButton.textContent = "再調整";
    updateRuntimeControls();
  }
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

function renderResult(result, startedAt) {
  promptOutput.value = result.distilled_prompt || "";
  const displayedElapsedMs = Number.isFinite(startedAt)
    ? performance.now() - startedAt
    : result.metrics?.latency_ms;
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
  compressionLatency.textContent = formatLatencySeconds(displayedElapsedMs);
  resultNotice.hidden = !result.should_send_original;
  resultNoticeDetail.textContent = result.should_send_original
    ? fallbackNoticeDetail(result.fallback_reason)
    : "";
  copyButton.textContent = result.should_send_original ? "原文をコピー" : "コピー";
}

function clearResultLists() {
  promptOutput.value = "";
  tokenComparison.textContent = "-";
  tokenRatioValue.textContent = "-";
  characterComparison.textContent = "-";
  characterRatioValue.textContent = "-";
  fallbackValue.textContent = "-";
  compressionLatency.textContent = "";
  resultNotice.hidden = true;
  resultNoticeDetail.textContent = "";
  copyButton.textContent = "コピー";
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
  return `${seconds.toFixed(1)}s`;
}

function setLoading(isLoading) {
  compressButton.disabled = isLoading || !modelInstalled || !modelRuntimeReady;
  clearInputButton.disabled = isLoading;
  compressButton.textContent = isLoading ? "圧縮中" : "圧縮";
}

function updateCompressButtonState() {
  compressButton.disabled = isCompressing || !modelInstalled || !modelRuntimeReady;
}

function setModelStatus(text, state) {
  updateStatusPill(modelStatus, "モデル", text, state);
}

function setModelReadyStatus() {
  setModelStatus("準備完了", "");
}

function setWorkStatus(text, state) {
  updateStatusPill(workStatus, "作業", text, state);
}

function updateStatusPill(element, label, text, state) {
  const fullStatus = `${label}: ${text}`;
  const labelElement = document.createElement("span");
  const textElement = document.createElement("span");
  labelElement.className = "status-label";
  labelElement.textContent = `${label}:`;
  textElement.className = "status-text";
  textElement.textContent = text;
  element.replaceChildren(labelElement, textElement);
  element.setAttribute("aria-label", fullStatus);
  element.title = fullStatus;
  element.className = `status-pill ${state || ""}`.trim();
}

async function copyTextToClipboard(text) {
  if (!text.trim()) {
    return false;
  }

  if (await copyTextWithNativeApi(text)) {
    return true;
  }

  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_error) {
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

function buildFallbackMessage(result) {
  const metrics = result.metrics || {};
  return `圧縮結果を安全に採用できなかったため原文を保持しました。自動コピーは行っていません。（文字数 ${formatComparison(metrics.input_characters, metrics.output_characters)}）`;
}

function fallbackNoticeDetail(reason) {
  if (typeof reason === "string" && reason.includes("model context")) {
    return "入力がモデルの文脈長を超えています。内容を分割して再実行してください。自動コピーは行っていません。";
  }
  return "要件保持を安全に確認できませんでした。内容を見直して再実行してください。自動コピーは行っていません。";
}

function summarizeText(text) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (compact.length <= 120) {
    return compact || "-";
  }
  return `${compact.slice(0, 117)}...`;
}

function notifyCompressionComplete(message, title = "圧縮完了") {
  return postDesktopNotification(title, message);
}

function showCompletionNotice(message, title) {
  notifyCompressionComplete(message, title);
}

async function refreshRuntimeStatus() {
  try {
    const response = await fetch("/api/runtime-status");
    if (!response.ok) {
      return;
    }
    const status = await response.json();
    if (status.profile && status.profile !== profileSelect.value) {
      return;
    }
    switch (status.phase) {
      case "checking":
        modelRuntimeReady = false;
        setModelStatus(status.message || "モデル確認中", "busy");
        break;
      case "downloading": {
        modelRuntimeReady = false;
        modelInstallInFlight = true;
        activeModelDownloadProfile = String(status.profile || profileSelect.value || "");
        setModelStatus(status.message || "モデル取得中", "busy");
        modelSetup.hidden = false;
        modelSetupTitle.textContent = "ローカルモデルを取得しています";
        const downloaded = Number(status.downloaded_bytes || 0);
        const total = Number(status.total_bytes || selectedModelStatus?.size_bytes || 0);
        modelSetupDetail.textContent = total > 0
          ? `${formatBytes(downloaded)} / ${formatBytes(total)}`
          : "Hugging Faceから取得しています。";
        modelDownloadProgress.hidden = false;
        modelDownloadProgress.value = total > 0 ? Math.min(100, (downloaded * 100) / total) : 0;
        modelCancelButton.hidden = false;
        modelCancelButton.disabled = modelCancelRequested;
        break;
      }
      case "cancelling":
        modelRuntimeReady = false;
        modelInstallInFlight = true;
        modelCancelRequested = true;
        activeModelDownloadProfile = String(status.profile || activeModelDownloadProfile);
        setModelStatus(status.message || "モデル取得を中止中", "busy");
        modelCancelButton.hidden = false;
        modelCancelButton.disabled = true;
        break;
      case "cancelled":
        modelRuntimeReady = false;
        modelInstallInFlight = false;
        modelCancelRequested = false;
        activeModelDownloadProfile = "";
        modelCancelButton.hidden = true;
        setModelStatus(status.message || "モデル取得中止", "");
        break;
      case "loading":
        modelRuntimeReady = false;
        modelCancelButton.hidden = true;
        setModelStatus(status.message || "モデル読み込み中", "busy");
        break;
      case "ready":
        modelRuntimeReady = true;
        modelInstallInFlight = false;
        modelCancelRequested = false;
        activeModelDownloadProfile = "";
        modelCancelButton.hidden = true;
        setModelReadyStatus();
        scheduleInputPrepare();
        break;
      case "error":
        modelRuntimeReady = false;
        modelInstallInFlight = false;
        modelCancelRequested = false;
        activeModelDownloadProfile = "";
        modelCancelButton.hidden = true;
        setModelStatus(status.message || "モデル読み込み失敗", "error");
        break;
      case "missing":
        modelRuntimeReady = false;
        setModelStatus(status.message || "モデル未取得", "error");
        break;
      case "skipped":
        modelRuntimeReady = true;
        setModelReadyStatus();
        scheduleInputPrepare();
        break;
      default:
        modelRuntimeReady = false;
        setModelStatus(status.message || "待機中", "");
        break;
    }
    updateCompressButtonState();
  } catch (_error) {
    // The UI can keep working even if the status probe races server startup.
  }
}

function updateSettingsSummary() {
  const profileLabel = profileSelect.value === "internal_llm" ? "内部モデル" : "自由選択";
  const levelDetail = compressionLevelDetails[levelInput.value];
  const levelLabel = levelDetail?.name || "標準";
  const summary = `${profileLabel} / ${levelLabel}`;
  settingsSummary.textContent = summary;
  settingsSummary.title = summary;
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
  for (const levelButton of levelButtons) {
    const isSelected = levelButton.dataset.compressionLevel === levelInput.value;
    levelButton.setAttribute("aria-pressed", String(isSelected));
  }
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

window.trimPromptOpenSettings = () => {
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
    const current = readStoredSettings(settingsStorageKey);
    const theme =
      localStorage.getItem(themeStorageKey) || localStorage.getItem(legacyThemeStorageKey);
    if (Object.keys(current).length > 0) {
      return normalizeLoadedSettings({ ...current, theme: current.theme || theme });
    }

    for (const legacyStorageKey of legacySettingsStorageKeys) {
      const legacy = readStoredSettings(legacyStorageKey);
      if (Object.keys(legacy).length === 0) {
        continue;
      }
      const migrated = legacy.profile === "lmstudio" ? { ...legacy, profile: undefined } : legacy;
      return normalizeLoadedSettings({ ...migrated, theme });
    }
    return normalizeLoadedSettings({ theme });
  } catch (_error) {
    return {};
  }
}

function readStoredSettings(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
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
    const settings = normalizeLoadedSettings(await response.json());
    if (Object.keys(settings).length > 0) {
      lastServerSettingsFingerprint = settingsFingerprint(settings);
    }
    return settings;
  } catch (_error) {
    return {};
  }
}

function normalizeLoadedSettings(settings) {
  const normalized = {};
  for (const key of [
    "profile",
    "level",
    "theme",
    "cpu_engine",
    "thread_mode",
    "generation_threads",
    "batch_threads"
  ]) {
    const value = settings[key];
    if (value !== undefined && value !== null && value !== "") {
      normalized[key] = value;
    }
  }
  return normalized;
}

function saveSettings() {
  const settings = collectSettings();
  const serialized = JSON.stringify(settings);
  if (serialized !== lastLocalSettingsJson) {
    localStorage.setItem(settingsStorageKey, serialized);
    localStorage.setItem(themeStorageKey, settings.theme);
    lastLocalSettingsJson = serialized;
  }
  scheduleSettingsSave(settings);
}

function collectSettings() {
  const generationThreads = positiveIntegerOrEmpty(generationThreadsInput.value);
  const batchThreads = positiveIntegerOrEmpty(batchThreadsInput.value);
  return {
    schema_version: 1,
    profile: profileSelect.value,
    level: Number(clampCompressionLevel(levelInput.value)),
    theme: themeToggle.checked ? "dark" : "light",
    cpu_engine: cpuEngineSelect.value || "auto",
    thread_mode: threadModeSelect.value === "manual" ? "manual" : "auto",
    generation_threads: generationThreads ? Number(generationThreads) : null,
    batch_threads: batchThreads ? Number(batchThreads) : null
  };
}

function settingsFingerprint(settings) {
  return JSON.stringify({
    profile: typeof settings.profile === "string" ? settings.profile : "",
    level: Number.isFinite(Number(settings.level)) ? Number(settings.level) : null,
    theme: settings.theme === "dark" ? "dark" : "light",
    cpu_engine: settings.cpu_engine || "auto",
    thread_mode: settings.thread_mode === "manual" ? "manual" : "auto",
    generation_threads: Number.isInteger(Number(settings.generation_threads))
      ? Number(settings.generation_threads)
      : null,
    batch_threads: Number.isInteger(Number(settings.batch_threads))
      ? Number(settings.batch_threads)
      : null
  });
}

function scheduleSettingsSave(settings) {
  const fingerprint = settingsFingerprint(settings);
  if (fingerprint === lastServerSettingsFingerprint) {
    pendingServerSettings = null;
    if (settingsSaveTimerId) {
      clearTimeout(settingsSaveTimerId);
      settingsSaveTimerId = 0;
    }
    return;
  }

  pendingServerSettings = { settings, fingerprint };
  if (settingsSaveTimerId) {
    clearTimeout(settingsSaveTimerId);
  }
  settingsSaveTimerId = setTimeout(flushSettingsSave, settingsSaveDelayMs);
}

async function flushSettingsSave() {
  settingsSaveTimerId = 0;
  if (settingsSaveInFlight || !pendingServerSettings) {
    return;
  }

  const pending = pendingServerSettings;
  pendingServerSettings = null;
  settingsSaveInFlight = true;
  const saved = await saveSettingsToServer(pending.settings);
  settingsSaveInFlight = false;
  if (saved) {
    lastServerSettingsFingerprint = pending.fingerprint;
  }

  if (pendingServerSettings?.fingerprint === lastServerSettingsFingerprint) {
    pendingServerSettings = null;
  } else if (pendingServerSettings && !settingsSaveTimerId) {
    settingsSaveTimerId = setTimeout(flushSettingsSave, settingsSaveDelayMs);
  }
}

async function saveSettingsToServer(settings) {
  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(settings)
    });
    return response.ok;
  } catch (_error) {
    // localStorage remains as a fallback when the local API is unavailable.
    return false;
  }
}

initializeTheme();
renderCompressionLevel();
updateCompressButtonState();

loadProfiles()
  .then(async () => {
    clearResultLists();
    if (modelInstalled) {
      const restarting = await runRuntimeSetupGate();
      if (restarting) {
        return;
      }
    } else {
      revealApplication();
    }
    refreshRuntimeStatus();
    setInterval(refreshRuntimeStatus, 1500);
  })
  .catch((error) => {
    showRuntimeSetupError(
      "TrimPromptを起動できませんでした",
      String(error.message || error)
    );
  });

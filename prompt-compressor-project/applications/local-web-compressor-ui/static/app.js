const profileSelect = document.querySelector("#profileSelect");
const modeSelect = document.querySelector("#modeSelect");
const taskSelect = document.querySelector("#taskSelect");
const levelInput = document.querySelector("#levelInput");
const levelValue = document.querySelector("#levelValue");
const compressButton = document.querySelector("#compressButton");
const sampleButton = document.querySelector("#sampleButton");
const copyButton = document.querySelector("#copyButton");
const promptInput = document.querySelector("#promptInput");
const promptOutput = document.querySelector("#promptOutput");
const runtimeStatus = document.querySelector("#runtimeStatus");
const inputTokens = document.querySelector("#inputTokens");
const outputTokens = document.querySelector("#outputTokens");
const ratioValue = document.querySelector("#ratioValue");
const fallbackValue = document.querySelector("#fallbackValue");
const preservedList = document.querySelector("#preservedList");
const riskList = document.querySelector("#riskList");
const removedList = document.querySelector("#removedList");
const settingsSummary = document.querySelector("#settingsSummary");

const modeLabels = {
  codex_optimized: "Codex optimized",
  developer_mode: "Developer mode",
  instruction_extract: "Instruction extract",
  lossless: "Lossless",
  privacy_redaction: "Privacy redaction"
};

const taskLabels = {
  coding: "Coding",
  log_analysis: "Log analysis",
  refactor: "Refactor",
  design_discussion: "Design discussion",
  general: "General"
};

const samplePrompt = [
  "Reactで作っている管理画面の一覧ページがあります。",
  "キーワード入力とステータス選択は、検索ボタンを押したときだけAPI検索してください。",
  "ページ番号変更時は従来通り即検索で大丈夫です。",
  "URLクエリへの検索条件保持は維持してください。",
  "TypeScriptで、既存のuseSearchParamsを使い、大きなリファクタは避けてください。"
].join("\n");

levelInput.addEventListener("input", () => {
  levelValue.textContent = levelInput.value;
  saveSettings();
  updateSettingsSummary();
});

profileSelect.addEventListener("change", () => {
  saveSettings();
  updateSettingsSummary();
});
modeSelect.addEventListener("change", () => {
  saveSettings();
  updateSettingsSummary();
});
taskSelect.addEventListener("change", () => {
  saveSettings();
  updateSettingsSummary();
});

sampleButton.addEventListener("click", () => {
  promptInput.value = samplePrompt;
  promptInput.focus();
});

copyButton.addEventListener("click", async () => {
  if (!promptOutput.value.trim()) {
    return;
  }
  await navigator.clipboard.writeText(promptOutput.value);
  setStatus("コピー済み", "");
});

compressButton.addEventListener("click", async () => {
  const inputText = promptInput.value.trim();
  if (!inputText) {
    setStatus("入力待ち", "error");
    promptInput.focus();
    return;
  }

  setLoading(true);
  setStatus("圧縮中", "busy");

  try {
    const response = await fetch("/api/compress", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        input_text: inputText,
        profile: profileSelect.value,
        task_type: taskSelect.value,
        compression_mode: modeSelect.value,
        compression_level: Number(levelInput.value),
        constraints: {
          preserve_code_blocks: true,
          preserve_file_names: true,
          preserve_error_messages: true,
          preserve_numbers: true,
          preserve_negations: true
        }
      })
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Compression failed");
    }

    renderResult(payload);
    setStatus(payload.should_send_original ? "原文推奨" : "完了", payload.should_send_original ? "error" : "");
  } catch (error) {
    setStatus("エラー", "error");
    promptOutput.value = String(error.message || error);
    clearResultLists();
  } finally {
    setLoading(false);
  }
});

async function loadProfiles() {
  const response = await fetch("/api/profiles");
  const payload = await response.json();
  const saved = loadSettings();
  const profiles = payload.profiles || [];
  const preferredProfile = profiles.some((profile) => profile.id === saved.profile)
    ? saved.profile
    : "standard";
  profileSelect.textContent = "";
  for (const profile of profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = `${profile.label} (${profile.model_ref})`;
    if (profile.id === preferredProfile) {
      option.selected = true;
    }
    profileSelect.append(option);
  }
  if (saved.mode) {
    modeSelect.value = saved.mode;
  }
  if (saved.task) {
    taskSelect.value = saved.task;
  }
  if (saved.level) {
    levelInput.value = saved.level;
    levelValue.textContent = saved.level;
  }
  updateSettingsSummary();
}

function renderResult(result) {
  promptOutput.value = result.distilled_prompt || "";
  const metrics = result.metrics || {};
  inputTokens.textContent = valueOrDash(metrics.input_tokens_est);
  outputTokens.textContent = valueOrDash(metrics.output_tokens_est);
  ratioValue.textContent = Number.isFinite(metrics.compression_ratio)
    ? `${Math.round(metrics.compression_ratio * 100)}%`
    : "-";
  fallbackValue.textContent = result.should_send_original ? "はい" : "いいえ";

  renderList(
    preservedList,
    result.preserved_requirements,
    (item) => `${item.kind}: ${item.text}`
  );
  renderList(
    riskList,
    result.risk_flags,
    (item) => `${item.code}: ${item.message}`,
    (item) => `risk-${item.severity || "medium"}`
  );
  renderList(removedList, result.removed_content_summary, (item) => item);

  if (result.fallback_reason) {
    const item = document.createElement("li");
    item.className = "risk-high";
    item.textContent = `理由: ${result.fallback_reason}`;
    riskList.append(item);
  }
}

function renderList(target, items, label, className) {
  target.textContent = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "なし";
    target.append(item);
    return;
  }

  for (const entry of items) {
    const item = document.createElement("li");
    item.textContent = label(entry);
    if (className) {
      item.className = className(entry);
    }
    target.append(item);
  }
}

function clearResultLists() {
  promptOutput.value = "";
  inputTokens.textContent = "-";
  outputTokens.textContent = "-";
  ratioValue.textContent = "-";
  fallbackValue.textContent = "-";
  renderList(preservedList, [], (item) => item);
  renderList(riskList, [], (item) => item);
  renderList(removedList, [], (item) => item);
}

function valueOrDash(value) {
  return Number.isFinite(value) ? String(value) : "-";
}

function setLoading(isLoading) {
  compressButton.disabled = isLoading;
  compressButton.textContent = isLoading ? "圧縮中" : "圧縮する";
}

function setStatus(text, state) {
  runtimeStatus.textContent = text;
  runtimeStatus.className = `status-pill ${state || ""}`.trim();
}

function updateSettingsSummary() {
  const profileLabel =
    profileSelect.selectedOptions[0]?.textContent || "Standard";
  const modeLabel = modeLabels[modeSelect.value] || modeSelect.value;
  const taskLabel = taskLabels[taskSelect.value] || taskSelect.value;
  settingsSummary.textContent = `${profileLabel} / ${modeLabel} / ${taskLabel} / Level ${levelInput.value}`;
}

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem("promptRelaySettings") || "{}");
  } catch (_error) {
    return {};
  }
}

function saveSettings() {
  localStorage.setItem(
    "promptRelaySettings",
    JSON.stringify({
      profile: profileSelect.value,
      mode: modeSelect.value,
      task: taskSelect.value,
      level: levelInput.value
    })
  );
}

loadProfiles()
  .then(() => {
    clearResultLists();
    setStatus("待機中", "");
  })
  .catch((error) => {
    setStatus("エラー", "error");
    promptOutput.value = String(error.message || error);
  });

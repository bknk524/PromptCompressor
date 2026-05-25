use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{CompressionError, Result};
use crate::profile::ProfileDefinition;
use crate::types::CompressionRequest;

#[derive(Debug, Clone)]
pub struct CompressionDraft {
    pub distilled_prompt: String,
    pub removed_content_summary: Vec<String>,
}

pub trait RuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRuntimeBackend;

impl RuntimeBackend for NoopRuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft> {
        let summary = vec![format!(
            "No runtime backend is connected yet; returning passthrough output for profile '{}'.",
            profile.id
        )];

        Ok(CompressionDraft {
            distilled_prompt: request.input_text.trim().to_string(),
            removed_content_summary: summary,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppProcessBackend {
    project_root: PathBuf,
    prompts_dir: PathBuf,
    models: ModelRegistry,
    runtimes: RuntimeRegistry,
}

impl LlamaCppProcessBackend {
    pub fn from_settings_dir(settings_dir: impl AsRef<Path>) -> Result<Self> {
        let settings_dir = settings_dir.as_ref();
        let project_root = settings_dir
            .parent()
            .ok_or_else(|| {
                CompressionError::InvalidConfig(format!(
                    "settings directory has no parent: {}",
                    settings_dir.display()
                ))
            })?
            .to_path_buf();

        Ok(Self {
            project_root: project_root.clone(),
            prompts_dir: project_root.join("prompt-templates"),
            models: ModelRegistry::from_path(settings_dir.join("model-catalog.yaml"))?,
            runtimes: RuntimeRegistry::from_path(settings_dir.join("runtime-backends.yaml"))?,
        })
    }

    fn resolve_model_and_runtime(
        &self,
        profile: &ProfileDefinition,
    ) -> Result<(&ModelDefinition, &RuntimeDefinition)> {
        let model = self.models.resolve(&profile.model_ref)?;
        let runtime = self.runtimes.resolve(&profile.runtime_ref)?;

        if model.runtime_ref != runtime.id {
            return Err(CompressionError::InvalidConfig(format!(
                "model '{}' uses runtime '{}', but profile '{}' points to '{}'",
                model.id, model.runtime_ref, profile.id, runtime.id
            )));
        }

        Ok((model, runtime))
    }

    fn build_llama_cpp_command(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<(Command, u64)> {
        let executable_path = runtime.executable_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing executable_path for llama.cpp",
                runtime.id
            ))
        })?;
        let executable_path = resolve_project_path(&self.project_root, executable_path);
        let executable_path = resolve_windows_exe(&executable_path);
        if !executable_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "llama.cpp executable not found at {}",
                executable_path.display()
            )));
        }

        let model_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for llama.cpp runtime '{}'",
                model.id, runtime.id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, model_path);
        if !model_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "model file not found at {}",
                model_path.display()
            )));
        }

        let prompt = self.build_prompt(request, profile, model)?;
        let mut command = Command::new(executable_path);
        command.current_dir(&self.project_root);
        command.arg("-m").arg(model_path);
        command.arg("-p").arg(prompt);
        command.arg("-n").arg(model.default_max_output.to_string());
        command.arg("--temp").arg("0");
        command
            .arg("--ctx-size")
            .arg(model.context_length.to_string());

        if runtime.threads != "auto" {
            command.arg("--threads").arg(&runtime.threads);
        }

        if runtime.gpu_layers > 0 {
            command
                .arg("--n-gpu-layers")
                .arg(runtime.gpu_layers.to_string());
        }

        Ok((command, runtime.timeout_ms))
    }

    fn compress_with_lmstudio(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for LM Studio",
                runtime.id
            ))
        })?;
        let model_name = self.resolve_lmstudio_model_name(model, runtime)?;
        let prompt = self.build_prompt(request, profile, model)?;
        let payload = ChatCompletionRequest {
            model: model_name.as_str(),
            messages: vec![ChatMessage {
                role: "user",
                content: prompt.as_str(),
            }],
            temperature: 0.0,
            max_tokens: model.default_max_output,
            stream: false,
        };
        let body = serde_json::to_vec(&payload).map_err(|error| {
            CompressionError::Runtime(format!("failed to serialize LM Studio request: {error}"))
        })?;
        let response_body = http_json_request(
            "POST",
            base_url,
            "/chat/completions",
            runtime.api_token_env.as_deref(),
            Some(&body),
            Duration::from_millis(runtime.timeout_ms),
        )?;
        let completion: ChatCompletionResponse =
            serde_json::from_str(&response_body).map_err(|error| {
                CompressionError::Runtime(format!(
                    "LM Studio response was not valid chat completion JSON: {error}"
                ))
            })?;
        let content = completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .ok_or_else(|| {
                CompressionError::Runtime(
                    "LM Studio response did not include choices[0].message.content".into(),
                )
            })?;

        parse_compression_output(content)
    }

    fn resolve_lmstudio_model_name(
        &self,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<String> {
        let configured = model.api_model.as_deref().unwrap_or(model.id.as_str());
        if configured != "auto" {
            return Ok(configured.to_string());
        }

        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for LM Studio",
                runtime.id
            ))
        })?;
        let response_body = http_json_request(
            "GET",
            base_url,
            "/models",
            runtime.api_token_env.as_deref(),
            None,
            Duration::from_millis(runtime.timeout_ms),
        )?;
        let models: ModelsResponse = serde_json::from_str(&response_body).map_err(|error| {
            CompressionError::Runtime(format!(
                "LM Studio /models response was not valid JSON: {error}"
            ))
        })?;

        models
            .data
            .first()
            .map(|item| item.id.clone())
            .ok_or_else(|| {
                CompressionError::Runtime(
                    "LM Studio returned no available models from /v1/models".into(),
                )
            })
    }

    fn build_prompt(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
    ) -> Result<String> {
        let template_path = self
            .prompts_dir
            .join(format!("{}.md", model.prompt_template));
        let template = fs::read_to_string(&template_path)?;
        let metadata = serde_json::json!({
            "task_type": &request.task_type,
            "compression_mode": &request.compression_mode,
            "compression_level": request.compression_level.value(),
            "profile": &profile.id,
            "policy_ref": &profile.policy_ref,
            "model_ref": &model.id,
            "adapter": &model.adapter,
            "thinking": model.thinking,
            "constraints": &request.constraints,
        });

        Ok(format!(
            "{template}\n\n\
             Return JSON with exactly these keys:\n\
             - distilled_prompt: string\n\
             - preserved_requirements: array of strings\n\
             - removed_content_summary: array of strings\n\
             - risk_flags: array of strings\n\
             - should_send_original: boolean\n\n\
             Request metadata:\n```json\n{}\n```\n\n\
             User request:\n```text\n{}\n```\n",
            serde_json::to_string_pretty(&metadata).map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to serialize runtime prompt metadata: {error}"
                ))
            })?,
            request.input_text.trim()
        ))
    }
}

impl RuntimeBackend for LlamaCppProcessBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match runtime.backend_kind.as_str() {
            "llama.cpp" => self.compress_with_llama_cpp(request, profile, model, runtime),
            "lmstudio" | "lm_studio" | "lm-studio" => {
                self.compress_with_lmstudio(request, profile, model, runtime)
            }
            other => Err(CompressionError::Runtime(format!(
                "unsupported runtime backend '{other}' for runtime '{}'",
                runtime.id
            ))),
        }
    }
}

impl LlamaCppProcessBackend {
    fn compress_with_llama_cpp(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        let (command, timeout_ms) =
            self.build_llama_cpp_command(request, profile, model, runtime)?;
        let output = run_command_with_timeout(command, Duration::from_millis(timeout_ms))?;

        if !output.status.success() {
            return Err(CompressionError::Runtime(format!(
                "llama.cpp process exited with status {}; stderr was {} bytes",
                output.status,
                output.stderr.len()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_compression_output(&stdout)
    }
}

#[derive(Debug, Clone)]
struct ModelRegistry {
    models: BTreeMap<String, ModelDefinition>,
}

impl ModelRegistry {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: ModelsFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported models schema_version: {}",
                file.schema_version
            )));
        }

        let models = file
            .models
            .into_iter()
            .map(|(id, entry)| {
                (
                    id.clone(),
                    ModelDefinition {
                        id,
                        label: entry.label,
                        adapter: entry.adapter,
                        runtime_ref: entry.runtime_ref,
                        model_path: entry.model_path.map(PathBuf::from),
                        api_model: entry.api_model,
                        quantization: entry.quantization,
                        context_length: entry.context_length,
                        thinking: entry.thinking,
                        default_max_output: entry.default_max_output,
                        prompt_template: entry.prompt_template,
                    },
                )
            })
            .collect();

        Ok(Self { models })
    }

    fn resolve(&self, id: &str) -> Result<&ModelDefinition> {
        self.models
            .get(id)
            .ok_or_else(|| CompressionError::UnknownModel(id.to_string()))
    }
}

#[derive(Debug, Clone)]
struct RuntimeRegistry {
    runtimes: BTreeMap<String, RuntimeDefinition>,
}

impl RuntimeRegistry {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: RuntimesFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported runtimes schema_version: {}",
                file.schema_version
            )));
        }

        let runtimes = file
            .runtimes
            .into_iter()
            .map(|(id, entry)| {
                (
                    id.clone(),
                    RuntimeDefinition {
                        id,
                        backend_kind: entry.backend_kind,
                        executable_path: entry.executable_path.map(PathBuf::from),
                        base_url: entry.base_url,
                        api_token_env: entry.api_token_env,
                        threads: entry.threads,
                        gpu_layers: entry.gpu_layers,
                        timeout_ms: entry.timeout_ms,
                    },
                )
            })
            .collect();

        Ok(Self { runtimes })
    }

    fn resolve(&self, id: &str) -> Result<&RuntimeDefinition> {
        self.runtimes
            .get(id)
            .ok_or_else(|| CompressionError::UnknownRuntime(id.to_string()))
    }
}

#[derive(Debug, Clone)]
struct ModelDefinition {
    id: String,
    #[allow(dead_code)]
    label: String,
    adapter: String,
    runtime_ref: String,
    model_path: Option<PathBuf>,
    api_model: Option<String>,
    #[allow(dead_code)]
    quantization: String,
    context_length: u32,
    thinking: bool,
    default_max_output: u32,
    prompt_template: String,
}

#[derive(Debug, Clone)]
struct RuntimeDefinition {
    id: String,
    backend_kind: String,
    executable_path: Option<PathBuf>,
    base_url: Option<String>,
    api_token_env: Option<String>,
    threads: String,
    gpu_layers: u32,
    timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
struct ModelsFile {
    schema_version: u32,
    models: BTreeMap<String, ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    label: String,
    adapter: String,
    #[serde(rename = "runtime")]
    runtime_ref: String,
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    api_model: Option<String>,
    #[serde(default)]
    quantization: String,
    #[serde(default = "default_context_length")]
    context_length: u32,
    #[serde(default)]
    thinking: bool,
    #[serde(default = "default_max_output")]
    default_max_output: u32,
    prompt_template: String,
}

#[derive(Debug, Deserialize)]
struct RuntimesFile {
    schema_version: u32,
    runtimes: BTreeMap<String, RuntimeEntry>,
}

#[derive(Debug, Deserialize)]
struct RuntimeEntry {
    #[serde(rename = "backend")]
    backend_kind: String,
    #[serde(default)]
    executable_path: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_token_env: Option<String>,
    #[serde(default = "default_threads")]
    threads: String,
    #[serde(default)]
    gpu_layers: u32,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

fn default_context_length() -> u32 {
    32768
}

fn default_max_output() -> u32 {
    256
}

fn default_threads() -> String {
    "auto".to_string()
}

fn default_timeout_ms() -> u64 {
    12000
}

#[derive(Debug, Deserialize)]
struct ModelCompressionOutput {
    distilled_prompt: String,
    #[serde(default)]
    removed_content_summary: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelListItem>,
}

#[derive(Debug, Deserialize)]
struct ModelListItem {
    id: String,
}

struct ProcessRunOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

struct HttpBaseUrl {
    host: String,
    port: u16,
    path_prefix: String,
}

fn http_json_request(
    method: &str,
    base_url: &str,
    endpoint_path: &str,
    api_token_env: Option<&str>,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<String> {
    let base = parse_http_base_url(base_url)?;
    let path = join_http_paths(&base.path_prefix, endpoint_path);
    let mut stream = TcpStream::connect((base.host.as_str(), base.port)).map_err(|error| {
        CompressionError::Runtime(format!(
            "failed to connect to LM Studio at {}:{}: {error}",
            base.host, base.port
        ))
    })?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let body = body.unwrap_or(&[]);
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\n\
         Host: {}\r\n\
         Accept: application/json\r\n\
         Connection: close\r\n",
        base.host
    );
    if let Some(token) = resolve_api_token(api_token_env) {
        request.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if !body.is_empty() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");

    stream.write_all(request.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    parse_http_response(&response)
}

fn parse_http_base_url(base_url: &str) -> Result<HttpBaseUrl> {
    let without_scheme = base_url.strip_prefix("http://").ok_or_else(|| {
        CompressionError::InvalidConfig(format!(
            "LM Studio base_url must use http:// for the local server: {base_url}"
        ))
    })?;
    let (host_port, path_prefix) = without_scheme
        .split_once('/')
        .map(|(host_port, path)| (host_port, format!("/{path}")))
        .unwrap_or((without_scheme, String::new()));
    let (host, port) = if let Some((host, port)) = host_port.rsplit_once(':') {
        let parsed_port = port.parse::<u16>().map_err(|error| {
            CompressionError::InvalidConfig(format!(
                "invalid LM Studio base_url port '{port}': {error}"
            ))
        })?;
        (host.to_string(), parsed_port)
    } else {
        (host_port.to_string(), 80)
    };

    if host.is_empty() {
        return Err(CompressionError::InvalidConfig(format!(
            "LM Studio base_url is missing a host: {base_url}"
        )));
    }

    Ok(HttpBaseUrl {
        host,
        port,
        path_prefix: path_prefix.trim_end_matches('/').to_string(),
    })
}

fn join_http_paths(path_prefix: &str, endpoint_path: &str) -> String {
    let prefix = path_prefix.trim_end_matches('/');
    let endpoint = endpoint_path.trim_start_matches('/');
    if prefix.is_empty() {
        format!("/{endpoint}")
    } else {
        format!("{prefix}/{endpoint}")
    }
}

fn resolve_api_token(api_token_env: Option<&str>) -> Option<String> {
    api_token_env
        .and_then(|name| env::var(name).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn parse_http_response(response: &[u8]) -> Result<String> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| {
            CompressionError::Runtime("LM Studio returned a malformed HTTP response".into())
        })?;
    let headers = std::str::from_utf8(&response[..header_end]).map_err(|error| {
        CompressionError::Runtime(format!(
            "LM Studio response headers were not UTF-8: {error}"
        ))
    })?;
    let status = parse_http_status(headers)?;
    let body_bytes = &response[header_end + 4..];
    let body_bytes = if has_chunked_transfer_encoding(headers) {
        decode_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    if !(200..300).contains(&status) {
        return Err(CompressionError::Runtime(format!(
            "LM Studio returned HTTP {status}: {}",
            body.trim()
        )));
    }

    Ok(body)
}

fn parse_http_status(headers: &str) -> Result<u16> {
    let status_line = headers
        .lines()
        .next()
        .ok_or_else(|| CompressionError::Runtime("LM Studio response had no status line".into()))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| {
            CompressionError::Runtime(format!(
                "LM Studio response status was malformed: {status_line}"
            ))
        })?
        .parse::<u16>()
        .map_err(|error| {
            CompressionError::Runtime(format!("LM Studio response status was invalid: {error}"))
        })?;
    Ok(status)
}

fn has_chunked_transfer_encoding(headers: &str) -> bool {
    headers.lines().any(|line| {
        line.split_once(':')
            .map(|(name, value)| {
                name.eq_ignore_ascii_case("transfer-encoding")
                    && value.to_ascii_lowercase().contains("chunked")
            })
            .unwrap_or(false)
    })
}

fn decode_chunked_body(mut body: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();

    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| {
                CompressionError::Runtime("chunked LM Studio response was truncated".into())
            })?;
        let size_line = std::str::from_utf8(&body[..line_end]).map_err(|error| {
            CompressionError::Runtime(format!("chunk size was not UTF-8: {error}"))
        })?;
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|error| {
            CompressionError::Runtime(format!("invalid chunk size '{size_hex}': {error}"))
        })?;
        body = &body[line_end + 2..];

        if size == 0 {
            break;
        }
        if body.len() < size + 2 {
            return Err(CompressionError::Runtime(
                "chunked LM Studio response body was shorter than declared".into(),
            ));
        }

        decoded.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }

    Ok(decoded)
}

fn run_command_with_timeout(mut command: Command, timeout: Duration) -> Result<ProcessRunOutput> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to start llama.cpp process: {error}"))
        })?;
    let started_at = Instant::now();

    loop {
        if let Some(status) = child.try_wait().map_err(|error| {
            CompressionError::Runtime(format!("failed to poll llama.cpp process: {error}"))
        })? {
            let mut stdout = Vec::new();
            if let Some(mut pipe) = child.stdout.take() {
                pipe.read_to_end(&mut stdout)?;
            }

            let mut stderr = Vec::new();
            if let Some(mut pipe) = child.stderr.take() {
                pipe.read_to_end(&mut stderr)?;
            }

            return Ok(ProcessRunOutput {
                status,
                stdout,
                stderr,
            });
        }

        if started_at.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CompressionError::RuntimeTimeout(timeout.as_millis() as u64));
        }

        thread::sleep(Duration::from_millis(25));
    }
}

fn parse_compression_output(output: &str) -> Result<CompressionDraft> {
    let start = output
        .find('{')
        .ok_or_else(|| CompressionError::Runtime("llama.cpp output did not contain JSON".into()))?;
    let end = output
        .rfind('}')
        .ok_or_else(|| CompressionError::Runtime("llama.cpp output did not contain JSON".into()))?;
    let json = &output[start..=end];
    let parsed: ModelCompressionOutput = serde_json::from_str(json).map_err(|error| {
        CompressionError::Runtime(format!(
            "llama.cpp output was not valid compression JSON: {error}"
        ))
    })?;

    Ok(CompressionDraft {
        distilled_prompt: parsed.distilled_prompt.trim().to_string(),
        removed_content_summary: parsed.removed_content_summary,
    })
}

fn resolve_project_path(project_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_root.join(path)
    }
}

fn resolve_windows_exe(path: &Path) -> PathBuf {
    if cfg!(windows) && path.extension().is_none() && !path.exists() {
        let mut candidate = path.to_path_buf();
        candidate.set_extension("exe");
        candidate
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_compression_output;

    #[test]
    fn parses_json_from_plain_model_output() {
        let draft = parse_compression_output(
            r#"{"distilled_prompt":"Fix search behavior.","removed_content_summary":["trimmed background"]}"#,
        )
        .expect("valid compression JSON");

        assert_eq!(draft.distilled_prompt, "Fix search behavior.");
        assert_eq!(draft.removed_content_summary, ["trimmed background"]);
    }

    #[test]
    fn parses_json_surrounded_by_runtime_text() {
        let draft = parse_compression_output(
            "llama.cpp banner\n{\"distilled_prompt\":\"Keep URL query params.\"}\n",
        )
        .expect("valid embedded compression JSON");

        assert_eq!(draft.distilled_prompt, "Keep URL query params.");
        assert!(draft.removed_content_summary.is_empty());
    }
}

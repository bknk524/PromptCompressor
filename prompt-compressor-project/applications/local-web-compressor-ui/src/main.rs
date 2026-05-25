use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use prompt_compressor_core::{
    CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
    CompressionService, LlamaCppProcessBackend, ProfileRegistry, RequestSource, RequestTarget,
    TaskType,
};
use serde::Deserialize;

const INDEX_HTML: &str = include_str!("../static/index.html");
const STYLES_CSS: &str = include_str!("../static/styles.css");
const APP_JS: &str = include_str!("../static/app.js");

#[derive(Debug, Parser)]
#[command(name = "prompt-compressor-local-ui")]
#[command(about = "Development local UI for Prompt Compressor")]
struct Args {
    #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, value_name = "PORT", default_value_t = 8787)]
    port: u16,

    #[arg(long, value_name = "DIR")]
    settings_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    registry: ProfileRegistry,
    backend: LlamaCppProcessBackend,
}

#[derive(Debug, Deserialize)]
struct CompressPayload {
    input_text: String,
    profile: String,
    task_type: TaskType,
    compression_mode: CompressionMode,
    compression_level: u8,
    constraints: Option<CompressionConstraints>,
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let settings_dir = resolve_settings_dir(args.settings_dir.as_deref())?;
    let profiles_path = settings_dir.join("compression-profiles.yaml");
    let registry = ProfileRegistry::from_path(&profiles_path)
        .with_context(|| format!("failed to load profiles from {}", profiles_path.display()))?;
    let backend = LlamaCppProcessBackend::from_settings_dir(&settings_dir)
        .context("failed to initialize llama.cpp process backend")?;
    let state = Arc::new(AppState { registry, backend });

    let address = format!("{}:{}", args.host, args.port);
    let listener =
        TcpListener::bind(&address).with_context(|| format!("failed to bind {address}"))?;

    println!("Prompt Compressor local UI: http://{address}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = state.clone();
                if let Err(error) = handle_client(stream, &state) {
                    eprintln!("request failed: {error:#}");
                }
            }
            Err(error) => eprintln!("connection failed: {error}"),
        }
    }

    Ok(())
}

fn handle_client(mut stream: TcpStream, state: &AppState) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let request = read_request(&mut stream)?;
    let response = route_request(request, state);
    stream.write_all(&response)?;
    stream.flush()?;
    Ok(())
}

fn route_request(request: HttpRequest, state: &AppState) -> Vec<u8> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => http_response(200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/styles.css") => {
            http_response(200, "text/css; charset=utf-8", STYLES_CSS.as_bytes())
        }
        ("GET", "/app.js") => http_response(
            200,
            "application/javascript; charset=utf-8",
            APP_JS.as_bytes(),
        ),
        ("GET", "/api/profiles") => json_response(200, &profiles_json(&state.registry)),
        ("POST", "/api/compress") => match compress_from_body(&request.body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "error": error.to_string()
                }),
            ),
        },
        _ => json_response(
            404,
            &serde_json::json!({
                "error": "not found"
            }),
        ),
    }
}

fn compress_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: CompressPayload = serde_json::from_slice(body).context("invalid JSON request")?;
    let request = CompressionRequest {
        input_text: payload.input_text,
        task_type: payload.task_type,
        compression_mode: payload.compression_mode,
        compression_level: CompressionLevel::from_u8(payload.compression_level)
            .context("invalid compression level")?,
        profile: payload.profile,
        constraints: payload.constraints.unwrap_or_default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };

    let service = CompressionService::new(state.registry.clone(), state.backend.clone());
    let result = service.compress(request)?;
    Ok(serde_json::to_value(result)?)
}

fn profiles_json(registry: &ProfileRegistry) -> serde_json::Value {
    let profiles: Vec<_> = registry
        .list()
        .into_iter()
        .map(|profile| {
            serde_json::json!({
                "id": profile.id,
                "label": profile.label,
                "model_ref": profile.model_ref,
                "policy_ref": profile.policy_ref,
                "runtime_ref": profile.runtime_ref,
                "fallback_profile": profile.fallback_profile,
                "target_tokenizer_profile": profile.target_tokenizer_profile,
            })
        })
        .collect();

    serde_json::json!({
        "profiles": profiles
    })
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];

    loop {
        let bytes_read = stream.read(&mut chunk)?;
        if bytes_read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..bytes_read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let header_end = find_header_end(&buffer).context("malformed HTTP request")?;
    let headers = std::str::from_utf8(&buffer[..header_end]).context("request was not UTF-8")?;
    let mut lines = headers.lines();
    let request_line = lines.next().context("missing request line")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().context("missing method")?.to_string();
    let raw_path = request_parts.next().context("missing path")?;
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();
    let content_length = parse_content_length(headers);

    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        let bytes_read = stream.read(&mut chunk)?;
        if bytes_read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..bytes_read]);
    }
    body.truncate(content_length);

    Ok(HttpRequest { method, path, body })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

fn json_response(status: u16, value: &serde_json::Value) -> Vec<u8> {
    match serde_json::to_vec_pretty(value) {
        Ok(body) => http_response(status, "application/json; charset=utf-8", &body),
        Err(error) => http_response(
            500,
            "application/json; charset=utf-8",
            format!(r#"{{"error":"failed to serialize response: {error}"}}"#).as_bytes(),
        ),
    }
}

fn http_response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

fn resolve_settings_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    find_upward_settings_dir(&cwd).ok_or_else(|| {
        anyhow::anyhow!("could not find ./settings directory from {}", cwd.display())
    })
}

fn find_upward_settings_dir(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join("settings");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{CompressionError, Result};
use crate::types::CompressionRequest;

use super::http_client::http_json_request;
use super::output_parser::{compression_response_schema, parse_compression_output};
use super::{effective_max_output_tokens, CompressionDraft, ModelDefinition, RuntimeDefinition};

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
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

pub(super) fn request_openai_completion(
    request: &CompressionRequest,
    prompt: &str,
    model: &ModelDefinition,
    runtime: &RuntimeDefinition,
    base_url: &str,
    model_name: &str,
) -> Result<CompressionDraft> {
    let payload = ChatCompletionRequest {
        model: model_name,
        messages: vec![
            ChatMessage {
                role: "system",
                content: "返答は JSON オブジェクトだけにしてください。日本語入力には日本語で返答し、コード、識別子、API 名、ファイル名は原文のまま保持してください。",
            },
            ChatMessage {
                role: "user",
                content: prompt,
            },
        ],
        temperature: 0.0,
        max_tokens: effective_max_output_tokens(request, model),
        stream: false,
        response_format: model
            .supports_json_schema
            .then(compression_response_schema),
    };
    let body = serde_json::to_vec(&payload).map_err(|error| {
        CompressionError::Runtime(format!(
            "failed to serialize local runtime request: {error}"
        ))
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
                "local runtime response was not valid chat completion JSON: {error}"
            ))
        })?;
    let content = completion
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .ok_or_else(|| {
            CompressionError::Runtime(
                "local runtime response did not include choices[0].message.content".into(),
            )
        })?;

    parse_compression_output(content)
}

pub(super) fn resolve_lmstudio_model_name(
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

#[cfg(test)]
mod tests {
    use super::ChatCompletionResponse;

    #[test]
    fn parses_first_chat_completion_content() {
        let response: ChatCompletionResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"{\"distilled_prompt\":\"短縮結果\"}"}}]}"#,
        )
        .expect("valid chat completion response");

        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some(r#"{"distilled_prompt":"短縮結果"}"#)
        );
    }
}

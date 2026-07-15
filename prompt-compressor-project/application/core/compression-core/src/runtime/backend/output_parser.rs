use serde::Deserialize;

use crate::error::{CompressionError, Result};

use super::CompressionDraft;

#[derive(Debug, Deserialize)]
struct ModelCompressionOutput {
    distilled_prompt: String,
    #[serde(default)]
    removed_content_summary: Vec<String>,
}

pub(super) fn compression_response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "compression_result",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "distilled_prompt": { "type": "string" },
                    "removed_content_summary": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["distilled_prompt", "removed_content_summary"],
                "additionalProperties": false
            }
        }
    })
}

pub(super) fn parse_compression_output(output: &str) -> Result<CompressionDraft> {
    let trimmed = output.trim();
    if let Ok(distilled_prompt) = serde_json::from_str::<String>(trimmed) {
        return Ok(CompressionDraft {
            distilled_prompt: clean_distilled_prompt_text(&distilled_prompt),
            removed_content_summary: Vec::new(),
        });
    }

    if trimmed.starts_with('"') && trimmed.contains("\"distilled_prompt\"") {
        let wrapped_json = format!("{{{trimmed}}}");
        if let Ok(parsed) = parse_model_compression_json(&wrapped_json) {
            return Ok(CompressionDraft {
                distilled_prompt: clean_distilled_prompt_text(&parsed.distilled_prompt),
                removed_content_summary: parsed.removed_content_summary,
            });
        }
    }

    let start = output.find('{').ok_or_else(|| {
        CompressionError::Runtime(format!(
            "llama.cpp output did not contain JSON; output starts with: {}",
            output_snippet(output)
        ))
    })?;
    let end = match output.rfind('}') {
        Some(end) => end,
        None => {
            if let Some(distilled_prompt) = extract_incomplete_distilled_prompt(output) {
                return Ok(CompressionDraft {
                    distilled_prompt: clean_distilled_prompt_text(&distilled_prompt),
                    removed_content_summary: Vec::new(),
                });
            }

            return Err(CompressionError::Runtime(format!(
                "llama.cpp output did not contain JSON; output starts with: {}",
                output_snippet(output)
            )));
        }
    };
    let json = &output[start..=end];
    let parsed = match parse_model_compression_json(json) {
        Ok(parsed) => parsed,
        Err(error) => {
            if let Some(parsed) = parse_first_valid_compression_json_object(output) {
                parsed
            } else {
                return Err(error);
            }
        }
    };

    Ok(CompressionDraft {
        distilled_prompt: clean_distilled_prompt_text(&parsed.distilled_prompt),
        removed_content_summary: parsed.removed_content_summary,
    })
}

fn parse_first_valid_compression_json_object(output: &str) -> Option<ModelCompressionOutput> {
    for (start, character) in output.char_indices() {
        if character != '{' {
            continue;
        }
        let candidate = &output[start..];
        let Some(end) = first_complete_json_object_end(candidate) else {
            continue;
        };
        if let Ok(parsed) = parse_model_compression_json(&candidate[..end]) {
            return Some(parsed);
        }
    }

    None
}

fn clean_distilled_prompt_text(text: &str) -> String {
    let mut cleaned = text.trim();
    for prefix in ["実行指示:", "実行指示：", "短縮文:", "短縮文："] {
        if let Some(stripped) = cleaned.strip_prefix(prefix) {
            cleaned = stripped.trim();
        }
    }
    for suffix in [
        ": 短縮文",
        "：短縮文",
        "： 短縮文",
        "; 短縮文",
        "；短縮文",
        "； 短縮文",
    ] {
        if let Some(stripped) = cleaned.strip_suffix(suffix) {
            cleaned = stripped.trim();
        }
    }
    cleaned.to_string()
}

fn parse_model_compression_json(json: &str) -> Result<ModelCompressionOutput> {
    match serde_json::from_str::<ModelCompressionOutput>(json) {
        Ok(parsed) => Ok(parsed),
        Err(primary_error) => {
            let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
                CompressionError::Runtime(format!(
                    "llama.cpp output was not valid JSON: {error}; output starts with: {}",
                    output_snippet(json)
                ))
            })?;

            if let Some(distilled_prompt) = extract_distilled_prompt_alias(&value) {
                return Ok(ModelCompressionOutput {
                    distilled_prompt,
                    removed_content_summary: extract_removed_summary_alias(&value),
                });
            }

            Err(CompressionError::Runtime(format!(
                "llama.cpp output was not valid compression JSON: {primary_error}; output starts with: {}",
                output_snippet(json)
            )))
        }
    }
}

fn extract_incomplete_distilled_prompt(output: &str) -> Option<String> {
    for key in [
        "distilled_prompt",
        "compressed_prompt",
        "compressed_text",
        "compressed",
        "output",
        "result",
        "prompt",
        "text",
        "summary",
    ] {
        let key_pattern = format!("\"{key}\"");
        let Some(key_start) = output.find(&key_pattern) else {
            continue;
        };
        let after_key = &output[key_start + key_pattern.len()..];
        let value_start = after_key.find(':')?;
        let mut value = after_key[value_start + 1..].trim_start();

        if let Some(stripped) = value.strip_prefix('"') {
            value = stripped;
            let mut text = String::new();
            let mut escaped = false;
            for character in value.chars() {
                if escaped {
                    text.push(match character {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '"' => '"',
                        '\\' => '\\',
                        other => other,
                    });
                    escaped = false;
                    continue;
                }

                match character {
                    '\\' => escaped = true,
                    '"' => break,
                    other => text.push(other),
                }
            }

            let text = text.trim().trim_end_matches('\\').trim();
            if !text.is_empty() {
                return Some(text.to_string());
            }
        } else {
            let text = value
                .split([',', '\n', '\r', '}'])
                .next()
                .unwrap_or_default()
                .trim()
                .trim_matches('"');
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    None
}

fn extract_distilled_prompt_alias(value: &serde_json::Value) -> Option<String> {
    const STRING_KEYS: &[&str] = &[
        "distilled_prompt",
        "compressed_prompt",
        "compressed_text",
        "compressed",
        "output",
        "result",
        "prompt",
        "text",
        "summary",
        "圧縮結果",
        "圧縮文",
        "短縮文",
        "要約",
    ];

    let object = value.as_object()?;
    for key in STRING_KEYS {
        if let Some(text) = object.get(*key).and_then(serde_json::Value::as_str) {
            let text = text.trim();
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    let string_values: Vec<_> = object
        .values()
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    (string_values.len() == 1).then(|| string_values[0].to_string())
}

fn extract_removed_summary_alias(value: &serde_json::Value) -> Vec<String> {
    const ARRAY_KEYS: &[&str] = &[
        "removed_content_summary",
        "removed_summary",
        "removed",
        "omitted",
        "削除内容",
    ];

    let Some(object) = value.as_object() else {
        return Vec::new();
    };

    for key in ARRAY_KEYS {
        if let Some(items) = object.get(*key).and_then(serde_json::Value::as_array) {
            return items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect();
        }
    }

    Vec::new()
}

pub(super) fn output_snippet(output: &str) -> String {
    output.chars().take(240).collect()
}

pub(super) fn first_complete_json_object_end(output: &str) -> Option<usize> {
    let start = output.find('{')?;
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, character) in output[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }

        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(start + offset + character.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{first_complete_json_object_end, parse_compression_output};

    #[test]
    fn finds_complete_object_after_runtime_prefix() {
        let output = "runtime: {\"distilled_prompt\":\"波括弧 } を保持\"} trailing";
        let end = first_complete_json_object_end(output).expect("complete object");

        assert_eq!(
            &output[..end],
            "runtime: {\"distilled_prompt\":\"波括弧 } を保持\"}"
        );
    }

    #[test]
    fn parses_alias_without_changing_the_distilled_text() {
        let draft = parse_compression_output("{\"compressed_text\":\"URLを保持する。\"}")
            .expect("supported alias");

        assert_eq!(draft.distilled_prompt, "URLを保持する。");
        assert!(draft.removed_content_summary.is_empty());
    }
}

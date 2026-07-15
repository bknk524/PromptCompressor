use std::env;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::error::{CompressionError, Result};

pub(super) struct HttpBaseUrl {
    pub(super) host: String,
    pub(super) port: u16,
    pub(super) path_prefix: String,
}

pub(super) fn http_json_request(
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
            "failed to connect to local runtime at {}:{}: {error}",
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

    let response = read_http_response(&mut stream)?;
    parse_http_response(&response)
}

fn read_http_response(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let bytes_read = stream.read(&mut buffer)?;
        if bytes_read == 0 {
            return Err(CompressionError::Runtime(
                "local runtime closed the connection before sending HTTP headers".into(),
            ));
        }
        response.extend_from_slice(&buffer[..bytes_read]);

        if let Some(header_end) = find_http_header_end(&response) {
            let headers = std::str::from_utf8(&response[..header_end]).map_err(|error| {
                CompressionError::Runtime(format!(
                    "local runtime response headers were not UTF-8: {error}"
                ))
            })?;

            if let Some(content_length) = http_content_length(headers)? {
                let expected_length = header_end + 4 + content_length;
                while response.len() < expected_length {
                    let bytes_read = stream.read(&mut buffer)?;
                    if bytes_read == 0 {
                        return Err(CompressionError::Runtime(
                            "local runtime closed the connection before the full response body arrived"
                                .into(),
                        ));
                    }
                    response.extend_from_slice(&buffer[..bytes_read]);
                }
                response.truncate(expected_length);
                return Ok(response);
            }

            if has_chunked_transfer_encoding(headers) {
                while !is_complete_chunked_body(&response[header_end + 4..])? {
                    let bytes_read = stream.read(&mut buffer)?;
                    if bytes_read == 0 {
                        return Err(CompressionError::Runtime(
                            "local runtime closed the connection before the chunked response completed"
                                .into(),
                        ));
                    }
                    response.extend_from_slice(&buffer[..bytes_read]);
                }
                return Ok(response);
            }

            stream.read_to_end(&mut response)?;
            return Ok(response);
        }
    }
}

fn find_http_header_end(response: &[u8]) -> Option<usize> {
    response.windows(4).position(|window| window == b"\r\n\r\n")
}

fn http_content_length(headers: &str) -> Result<Option<usize>> {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim())
        })
        .map(|value| {
            value.parse::<usize>().map_err(|error| {
                CompressionError::Runtime(format!(
                    "local runtime returned an invalid Content-Length '{value}': {error}"
                ))
            })
        })
        .transpose()
}

fn is_complete_chunked_body(mut body: &[u8]) -> Result<bool> {
    loop {
        let Some(line_end) = body.windows(2).position(|window| window == b"\r\n") else {
            return Ok(false);
        };
        let size_line = std::str::from_utf8(&body[..line_end]).map_err(|error| {
            CompressionError::Runtime(format!("chunk size was not UTF-8: {error}"))
        })?;
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|error| {
            CompressionError::Runtime(format!("invalid chunk size '{size_hex}': {error}"))
        })?;
        body = &body[line_end + 2..];

        if size == 0 {
            return Ok(true);
        }
        if body.len() < size + 2 {
            return Ok(false);
        }
        body = &body[size + 2..];
    }
}

pub(super) fn parse_http_base_url(base_url: &str) -> Result<HttpBaseUrl> {
    let without_scheme = base_url.strip_prefix("http://").ok_or_else(|| {
        CompressionError::InvalidConfig(format!(
            "local runtime base_url must use http:// for the local server: {base_url}"
        ))
    })?;
    let (host_port, path_prefix) = without_scheme
        .split_once('/')
        .map(|(host_port, path)| (host_port, format!("/{path}")))
        .unwrap_or((without_scheme, String::new()));
    let (host, port) = if let Some((host, port)) = host_port.rsplit_once(':') {
        let parsed_port = port.parse::<u16>().map_err(|error| {
            CompressionError::InvalidConfig(format!(
                "invalid local runtime base_url port '{port}': {error}"
            ))
        })?;
        (host.to_string(), parsed_port)
    } else {
        (host_port.to_string(), 80)
    };

    if host.is_empty() {
        return Err(CompressionError::InvalidConfig(format!(
            "local runtime base_url is missing a host: {base_url}"
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
    let header_end = find_http_header_end(response).ok_or_else(|| {
        CompressionError::Runtime("local runtime returned a malformed HTTP response".into())
    })?;
    let headers = std::str::from_utf8(&response[..header_end]).map_err(|error| {
        CompressionError::Runtime(format!(
            "local runtime response headers were not UTF-8: {error}"
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
            "local runtime returned HTTP {status}: {}",
            body.trim()
        )));
    }

    Ok(body)
}

fn parse_http_status(headers: &str) -> Result<u16> {
    let status_line = headers.lines().next().ok_or_else(|| {
        CompressionError::Runtime("local runtime response had no status line".into())
    })?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| {
            CompressionError::Runtime(format!(
                "local runtime response status was malformed: {status_line}"
            ))
        })?
        .parse::<u16>()
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "local runtime response status was invalid: {error}"
            ))
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
                CompressionError::Runtime("chunked local runtime response was truncated".into())
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
                "chunked local runtime response body was shorter than declared".into(),
            ));
        }

        decoded.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::{decode_chunked_body, parse_http_response};

    #[test]
    fn decodes_chunked_json_response() {
        let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"ok\":1\r\n1\r\n}\r\n0\r\n\r\n";

        assert_eq!(
            parse_http_response(response).expect("valid chunked response"),
            "{\"ok\":1}"
        );
    }

    #[test]
    fn rejects_shorter_than_declared_chunk() {
        let error = decode_chunked_body(b"9\r\nabc\r\n0\r\n\r\n")
            .expect_err("truncated chunk must be rejected");

        assert!(error.to_string().contains("shorter than declared"));
    }
}

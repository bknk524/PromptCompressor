use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{CompressionError, Result};

const MAX_MODEL_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const DOWNLOAD_BUFFER_BYTES: usize = 1024 * 1024;
const DOWNLOAD_CHUNK_BYTES: u64 = 32 * 1024 * 1024;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const VERIFICATION_STAMP_SCHEMA: u8 = 1;
const PARTIAL_DOWNLOAD_STAMP_SCHEMA: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationStamp {
    schema_version: u8,
    size_bytes: u64,
    modified_unix_nanos: u128,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PartialDownloadStamp {
    schema_version: u8,
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ModelDownloadCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ModelDownloadCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(super) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(CompressionError::Cancelled(
                "model download was cancelled".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub(super) struct ModelDownloadSpec {
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    size_bytes: u64,
}

impl ModelDownloadSpec {
    pub(super) fn new(
        repository: String,
        revision: String,
        filename: String,
        sha256: String,
        size_bytes: u64,
    ) -> Result<Self> {
        validate_repository(&repository)?;
        validate_revision(&revision)?;
        validate_filename(&filename)?;
        validate_sha256(&sha256)?;
        if size_bytes == 0 || size_bytes > MAX_MODEL_BYTES {
            return Err(CompressionError::InvalidConfig(format!(
                "Hugging Face model size_bytes must be between 1 and {MAX_MODEL_BYTES}"
            )));
        }

        Ok(Self {
            repository,
            revision,
            filename,
            sha256,
            size_bytes,
        })
    }

    pub(super) fn filename(&self) -> &str {
        &self.filename
    }

    pub(super) fn repository(&self) -> &str {
        &self.repository
    }

    pub(super) fn revision(&self) -> &str {
        &self.revision
    }

    pub(super) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn repository_parts(&self) -> (&str, &str) {
        self.repository
            .split_once('/')
            .expect("validated Hugging Face repository must contain one slash")
    }

    fn partial_stamp(&self) -> PartialDownloadStamp {
        PartialDownloadStamp {
            schema_version: PARTIAL_DOWNLOAD_STAMP_SCHEMA,
            repository: self.repository.clone(),
            revision: self.revision.clone(),
            filename: self.filename.clone(),
            sha256: self.sha256.clone(),
            size_bytes: self.size_bytes,
        }
    }
}

pub(super) fn ensure_model_file(
    destination: &Path,
    spec: &ModelDownloadSpec,
    cancellation: &ModelDownloadCancellation,
    progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
) -> Result<()> {
    if verify_existing_model(destination, spec)? {
        progress(ModelDownloadProgress {
            downloaded_bytes: spec.size_bytes,
            total_bytes: spec.size_bytes,
        });
        return Ok(());
    }

    let parent = destination.parent().ok_or_else(|| {
        CompressionError::InvalidConfig(format!(
            "model destination has no parent: {}",
            destination.display()
        ))
    })?;
    fs::create_dir_all(parent)?;

    let (partial_path, downloaded_bytes) = prepare_partial_download(destination, spec)?;
    progress(ModelDownloadProgress {
        downloaded_bytes,
        total_bytes: spec.size_bytes,
    });
    download_from_hugging_face(
        &partial_path,
        downloaded_bytes,
        spec,
        cancellation,
        progress,
    )?;
    cancellation.check()?;

    if let Err(error) = verify_downloaded_file(&partial_path, spec, Some(cancellation)) {
        if !matches!(error, CompressionError::Cancelled(_)) {
            discard_partial_download(destination)?;
        }
        return Err(error);
    }

    install_verified_downloaded_file(&partial_path, destination, spec)?;
    remove_file_if_exists(&partial_path)?;
    remove_file_if_exists(&partial_download_stamp_path(destination)?)?;
    Ok(())
}

fn download_from_hugging_face(
    partial_path: &Path,
    downloaded_bytes: u64,
    spec: &ModelDownloadSpec,
    cancellation: &ModelDownloadCancellation,
    progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
) -> Result<()> {
    let (owner, name) = spec.repository_parts();
    let client = HFClient::builder()
        .endpoint("https://huggingface.co")
        .cache_enabled(false)
        .user_agent("PromptCompressor/0.1")
        .retry_max_attempts(3)
        .build_sync()
        .map_err(hugging_face_error)?;
    let repository = client.model(owner, name);

    download_resumable_chunks(
        partial_path,
        downloaded_bytes,
        spec.size_bytes,
        DOWNLOAD_CHUNK_BYTES,
        cancellation,
        progress,
        |range| {
            repository
                .download_file_to_bytes()
                .filename(spec.filename.clone())
                .revision(spec.revision.clone())
                .range(range)
                .send()
                .map(|bytes| bytes.to_vec())
                .map_err(hugging_face_error)
        },
    )
}

fn download_resumable_chunks(
    partial_path: &Path,
    downloaded_bytes: u64,
    total_bytes: u64,
    chunk_bytes: u64,
    cancellation: &ModelDownloadCancellation,
    progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
    mut fetch: impl FnMut(Range<u64>) -> Result<Vec<u8>>,
) -> Result<()> {
    if chunk_bytes == 0 || downloaded_bytes > total_bytes {
        return Err(CompressionError::Runtime(
            "invalid resumable download state".into(),
        ));
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(partial_path)?;
    if file.metadata()?.len() != downloaded_bytes {
        return Err(CompressionError::Runtime(
            "partial model size changed before resume".into(),
        ));
    }

    let mut offset = downloaded_bytes;
    while offset < total_bytes {
        cancellation.check()?;
        let end = offset.saturating_add(chunk_bytes).min(total_bytes);
        let bytes = fetch(offset..end)?;
        let expected = usize::try_from(end - offset)
            .map_err(|_| CompressionError::Runtime("model download chunk is too large".into()))?;
        if bytes.len() != expected {
            return Err(CompressionError::Runtime(format!(
                "model download range size mismatch: expected {expected}, received {}",
                bytes.len()
            )));
        }
        file.write_all(&bytes)?;
        file.sync_data()?;
        offset = end;
        progress(ModelDownloadProgress {
            downloaded_bytes: offset,
            total_bytes,
        });
    }
    Ok(())
}

fn hugging_face_error(error: hf_hub::HFError) -> CompressionError {
    CompressionError::Runtime(format!(
        "failed to download model from Hugging Face: {error}"
    ))
}

pub(super) fn resumable_downloaded_bytes(
    destination: &Path,
    spec: &ModelDownloadSpec,
) -> Result<u64> {
    let partial_path = partial_download_path(destination)?;
    let metadata = match fs::symlink_metadata(&partial_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > spec.size_bytes
    {
        return Ok(0);
    }

    let stamp_path = partial_download_stamp_path(destination)?;
    let stamp: PartialDownloadStamp = match fs::read(&stamp_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(stamp) => stamp,
        None => return Ok(0),
    };
    if stamp != spec.partial_stamp() {
        return Ok(0);
    }
    Ok(metadata.len())
}

fn prepare_partial_download(
    destination: &Path,
    spec: &ModelDownloadSpec,
) -> Result<(PathBuf, u64)> {
    let partial_path = partial_download_path(destination)?;
    let downloaded_bytes = resumable_downloaded_bytes(destination, spec)?;
    if downloaded_bytes > 0 {
        return Ok((partial_path, downloaded_bytes));
    }

    discard_partial_download(destination)?;
    write_partial_download_stamp(destination, &spec.partial_stamp())?;
    File::create(&partial_path)?;
    Ok((partial_path, 0))
}

fn write_partial_download_stamp(destination: &Path, stamp: &PartialDownloadStamp) -> Result<()> {
    let path = partial_download_stamp_path(destination)?;
    let bytes = serde_json::to_vec(stamp).map_err(|error| {
        CompressionError::Runtime(format!(
            "failed to serialize partial model download state: {error}"
        ))
    })?;
    write_bytes_atomically(&path, &bytes)
}

fn discard_partial_download(destination: &Path) -> Result<()> {
    remove_file_if_exists(&partial_download_path(destination)?)?;
    remove_file_if_exists(&partial_download_stamp_path(destination)?)
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn partial_download_path(destination: &Path) -> Result<PathBuf> {
    append_filename_suffix(destination, ".partial")
}

fn partial_download_stamp_path(destination: &Path) -> Result<PathBuf> {
    append_filename_suffix(destination, ".partial.json")
}

#[cfg(test)]
fn install_downloaded_file(
    downloaded_path: &Path,
    destination: &Path,
    spec: &ModelDownloadSpec,
) -> Result<()> {
    verify_downloaded_file(downloaded_path, spec, None)?;
    install_verified_downloaded_file(downloaded_path, destination, spec)
}

fn verify_downloaded_file(
    downloaded_path: &Path,
    spec: &ModelDownloadSpec,
    cancellation: Option<&ModelDownloadCancellation>,
) -> Result<()> {
    let metadata = fs::metadata(downloaded_path)?;
    if !metadata.is_file() || metadata.len() != spec.size_bytes {
        return Err(CompressionError::Runtime(format!(
            "model download size mismatch: expected {}, received {}",
            spec.size_bytes,
            metadata.len()
        )));
    }
    let actual_sha256 = hash_file_with_cancellation(downloaded_path, cancellation)?;
    if actual_sha256 != spec.sha256 {
        return Err(CompressionError::Runtime(format!(
            "model download SHA-256 mismatch: expected {}, received {actual_sha256}",
            spec.sha256
        )));
    }
    Ok(())
}

fn install_verified_downloaded_file(
    downloaded_path: &Path,
    destination: &Path,
    spec: &ModelDownloadSpec,
) -> Result<()> {
    // 別プロセスが先に同じモデルを配置した場合は、検証済みの既存ファイルを優先する。
    if verify_existing_model(destination, spec)? {
        return Ok(());
    }
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(downloaded_path, destination)?;
    write_verification_stamp(destination, spec, &fs::metadata(destination)?)?;
    Ok(())
}

pub(super) fn verify_existing_model(destination: &Path, spec: &ModelDownloadSpec) -> Result<bool> {
    let metadata = match fs::metadata(destination) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() != spec.size_bytes {
        return Ok(false);
    }

    if verification_stamp_matches(destination, spec, &metadata)? {
        return Ok(true);
    }

    let actual_sha256 = hash_file(destination)?;
    if actual_sha256 != spec.sha256 {
        let _ = fs::remove_file(verification_stamp_path(destination)?);
        return Ok(false);
    }
    write_verification_stamp(destination, spec, &metadata)?;
    Ok(true)
}

fn verification_stamp_matches(
    destination: &Path,
    spec: &ModelDownloadSpec,
    metadata: &fs::Metadata,
) -> Result<bool> {
    let path = verification_stamp_path(destination)?;
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let stamp: VerificationStamp = match serde_json::from_slice(&bytes) {
        Ok(stamp) => stamp,
        Err(_) => return Ok(false),
    };
    Ok(stamp.schema_version == VERIFICATION_STAMP_SCHEMA
        && stamp.size_bytes == spec.size_bytes
        && stamp.sha256 == spec.sha256
        && stamp.modified_unix_nanos == modified_unix_nanos(metadata)?)
}

fn write_verification_stamp(
    destination: &Path,
    spec: &ModelDownloadSpec,
    metadata: &fs::Metadata,
) -> Result<()> {
    let stamp = VerificationStamp {
        schema_version: VERIFICATION_STAMP_SCHEMA,
        size_bytes: spec.size_bytes,
        modified_unix_nanos: modified_unix_nanos(metadata)?,
        sha256: spec.sha256.clone(),
    };
    let path = verification_stamp_path(destination)?;
    let bytes = serde_json::to_vec(&stamp).map_err(|error| {
        CompressionError::Runtime(format!("failed to serialize model verification: {error}"))
    })?;
    write_bytes_atomically(&path, &bytes)
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = append_filename_suffix(
        path,
        &format!(
            ".tmp-{}",
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ),
    )?;
    fs::write(&temporary, bytes)?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn verification_stamp_path(destination: &Path) -> Result<PathBuf> {
    append_filename_suffix(destination, ".verified.json")
}

fn modified_unix_nanos(metadata: &fs::Metadata) -> Result<u128> {
    metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "model modification time predates {:?}: {error}",
                SystemTime::UNIX_EPOCH
            ))
        })
}

fn hash_file(path: &Path) -> Result<String> {
    hash_file_with_cancellation(path, None)
}

fn hash_file_with_cancellation(
    path: &Path,
    cancellation: Option<&ModelDownloadCancellation>,
) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        if let Some(cancellation) = cancellation {
            cancellation.check()?;
        }
        let read_bytes = file.read(&mut buffer)?;
        if read_bytes == 0 {
            break;
        }
        hasher.update(&buffer[..read_bytes]);
    }
    Ok(encode_hex(&hasher.finalize()))
}

#[cfg(test)]
fn sha256_bytes(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn append_filename_suffix(path: &Path, suffix: &str) -> Result<PathBuf> {
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model path has no valid UTF-8 filename: {}",
                path.display()
            ))
        })?;
    Ok(path.with_file_name(format!("{filename}{suffix}")))
}

fn validate_repository(repository: &str) -> Result<()> {
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if parts.next().is_some()
        || !is_hugging_face_identifier(owner)
        || !is_hugging_face_identifier(name)
    {
        return Err(CompressionError::InvalidConfig(format!(
            "invalid Hugging Face repository: {repository}"
        )));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<()> {
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|value| value.is_ascii_digit() || matches!(value, b'a'..=b'f'))
    {
        return Err(CompressionError::InvalidConfig(
            "Hugging Face revision must be a 40-character lowercase commit SHA".into(),
        ));
    }
    Ok(())
}

fn validate_filename(filename: &str) -> Result<()> {
    if filename.len() > 255
        || !filename.ends_with(".gguf")
        || !is_hugging_face_identifier(filename)
        || Path::new(filename)
            .file_name()
            .and_then(|value| value.to_str())
            != Some(filename)
    {
        return Err(CompressionError::InvalidConfig(format!(
            "invalid Hugging Face GGUF filename: {filename}"
        )));
    }
    Ok(())
}

fn validate_sha256(sha256: &str) -> Result<()> {
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|value| value.is_ascii_digit() || matches!(value, b'a'..=b'f'))
    {
        return Err(CompressionError::InvalidConfig(
            "model sha256 must be 64 lowercase hexadecimal characters".into(),
        ));
    }
    Ok(())
}

fn is_hugging_face_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.bytes().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, b'.' | b'_' | b'-')
        })
        && value != "."
        && value != ".."
}

#[cfg(test)]
mod tests {
    use super::{
        download_resumable_chunks, install_downloaded_file, prepare_partial_download, sha256_bytes,
        verification_stamp_path, verify_existing_model, ModelDownloadCancellation,
        ModelDownloadSpec,
    };
    use crate::error::CompressionError;
    use hf_hub::HFClient;
    use std::fs;
    use std::ops::Range;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[test]
    fn rejects_unpinned_or_traversing_hugging_face_sources() {
        let valid_hash = "0".repeat(64);
        assert!(ModelDownloadSpec::new(
            "owner/model".to_string(),
            "main".to_string(),
            "model.gguf".to_string(),
            valid_hash.clone(),
            1,
        )
        .is_err());
        assert!(ModelDownloadSpec::new(
            "owner/model".to_string(),
            "a".repeat(40),
            "../model.gguf".to_string(),
            valid_hash,
            1,
        )
        .is_err());
    }

    #[test]
    fn installs_verified_model() {
        let directory = test_directory();
        let destination = directory.join("model.gguf");
        let downloaded_path = directory.join("downloaded.gguf");
        fs::create_dir_all(&directory).expect("test directory should be created");
        let bytes = b"verified model";
        let spec = test_spec(bytes);
        fs::write(&downloaded_path, bytes).expect("downloaded model should be written");

        install_downloaded_file(&downloaded_path, &destination, &spec)
            .expect("verified model should be installed");

        assert_eq!(fs::read(&destination).expect("model should exist"), bytes);
        assert!(!downloaded_path.exists());
        assert!(verification_stamp_path(&destination)
            .expect("verification path")
            .is_file());
        assert!(verify_existing_model(&destination, &spec).expect("cached verification"));
        fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn hash_mismatch_does_not_replace_existing_model() {
        let directory = test_directory();
        let destination = directory.join("model.gguf");
        let downloaded_path = directory.join("downloaded.gguf");
        fs::create_dir_all(&directory).expect("test directory should be created");
        fs::write(&destination, b"old!").expect("existing model should be written");
        fs::write(&downloaded_path, b"evil").expect("downloaded model should be written");
        let spec = test_spec(b"good");

        let result = install_downloaded_file(&downloaded_path, &destination, &spec);

        assert!(result.is_err());
        assert_eq!(
            fs::read(&destination).expect("existing model should remain"),
            b"old!"
        );
        fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn resumes_from_a_matching_partial_download() {
        let directory = test_directory();
        let destination = directory.join("model.gguf");
        fs::create_dir_all(&directory).expect("test directory should be created");
        let source = b"abcdefgh";
        let spec = test_spec(source);
        let (partial_path, offset) =
            prepare_partial_download(&destination, &spec).expect("prepare partial download");
        assert_eq!(offset, 0);
        fs::write(&partial_path, &source[..3]).expect("write resumable prefix");
        let (partial_path, offset) =
            prepare_partial_download(&destination, &spec).expect("resume partial download");
        let mut requested = Vec::<Range<u64>>::new();
        let mut progress = Vec::new();

        download_resumable_chunks(
            &partial_path,
            offset,
            source.len() as u64,
            2,
            &ModelDownloadCancellation::default(),
            &mut |value| progress.push(value.downloaded_bytes),
            |range| {
                requested.push(range.clone());
                Ok(source[range.start as usize..range.end as usize].to_vec())
            },
        )
        .expect("resumed chunks should download");

        assert_eq!(requested, [3..5, 5..7, 7..8]);
        assert_eq!(progress, [5, 7, 8]);
        assert_eq!(fs::read(partial_path).expect("partial model"), source);
        fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn cancellation_preserves_completed_chunks_for_resume() {
        let directory = test_directory();
        let destination = directory.join("model.gguf");
        fs::create_dir_all(&directory).expect("test directory should be created");
        let source = b"abcdefgh";
        let spec = test_spec(source);
        let (partial_path, offset) =
            prepare_partial_download(&destination, &spec).expect("prepare partial download");
        let cancellation = ModelDownloadCancellation::default();
        let cancel_after_first_chunk = cancellation.clone();

        let result = download_resumable_chunks(
            &partial_path,
            offset,
            source.len() as u64,
            2,
            &cancellation,
            &mut |_| {},
            |range| {
                cancel_after_first_chunk.cancel();
                Ok(source[range.start as usize..range.end as usize].to_vec())
            },
        );

        assert!(matches!(result, Err(CompressionError::Cancelled(_))));
        assert_eq!(fs::read(partial_path).expect("partial model"), &source[..2]);
        fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    fn changed_download_identity_discards_stale_partial_bytes() {
        let directory = test_directory();
        let destination = directory.join("model.gguf");
        fs::create_dir_all(&directory).expect("test directory should be created");
        let first_spec = test_spec(b"first");
        let (partial_path, _) =
            prepare_partial_download(&destination, &first_spec).expect("prepare first partial");
        fs::write(&partial_path, b"fir").expect("write stale partial");
        let replacement_spec = test_spec(b"other");

        let (partial_path, offset) = prepare_partial_download(&destination, &replacement_spec)
            .expect("prepare replacement partial");

        assert_eq!(offset, 0);
        assert_eq!(fs::metadata(partial_path).expect("fresh partial").len(), 0);
        fs::remove_dir_all(directory).expect("test directory should be removed");
    }

    #[test]
    #[ignore = "requires network access to Hugging Face"]
    fn downloads_sarashina_byte_range_through_hugging_face_xet() {
        let bytes = HFClient::builder()
            .endpoint("https://huggingface.co")
            .cache_enabled(false)
            .user_agent("PromptCompressor/0.1-test")
            .build_sync()
            .expect("Hugging Face client should be created")
            .model("mmnga", "sarashina2.2-3b-instruct-v0.1-gguf")
            .download_file_to_bytes()
            .filename("sarashina2.2-3b-instruct-v0.1-Q4_K_S.gguf")
            .revision("31d771319b04032f33e0d9d860f3984ea4812154")
            .range(0..1)
            .send()
            .expect("pinned Sarashina model byte should download");

        assert_eq!(bytes.as_ref(), b"G");
    }

    fn test_spec(bytes: &[u8]) -> ModelDownloadSpec {
        ModelDownloadSpec::new(
            "owner/model".to_string(),
            "a".repeat(40),
            "model.gguf".to_string(),
            sha256_bytes(bytes),
            bytes.len() as u64,
        )
        .expect("test model source should be valid")
    }

    fn test_directory() -> PathBuf {
        std::env::temp_dir().join(format!("prompt-compressor-model-test-{}", Uuid::new_v4()))
    }
}

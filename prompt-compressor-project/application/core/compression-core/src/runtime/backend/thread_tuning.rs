use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{CompressionError, Result};

use super::super::catalog::{ModelDefinition, RuntimeDefinition};
#[cfg(feature = "embedded-llama")]
use super::embedded_llama as llama_cpp;

const THREAD_TUNING_SCHEMA_VERSION: u32 = 1;
const THREAD_TUNING_CONTRACT_VERSION: u32 = 5;
const MAX_AUTOMATIC_RUNTIME_THREADS: usize = 8;
const BENCHMARK_CONTEXT_LENGTH: u32 = 1_024;
const BENCHMARK_WARMUP_TOKENS: usize = 8;
const BENCHMARK_BATCH_TOKENS: usize = 16;
const BENCHMARK_GENERATION_TOKENS: usize = 8;
const BENCHMARK_INITIAL_ROUNDS: usize = 3;
const BENCHMARK_UNSTABLE_EXTRA_ROUNDS: usize = 2;
const BENCHMARK_STABLE_PERCENT: u128 = 120;
const NEAR_FASTEST_PERCENT: u128 = 103;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct RuntimeThreadCounts {
    pub(super) generation: u32,
    pub(super) batch: u32,
}

#[derive(Debug)]
pub(super) struct ThreadTuningStore {
    directory: PathBuf,
    memory: Mutex<BTreeMap<String, RuntimeThreadCounts>>,
    pending_next_launch: Mutex<BTreeSet<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedThreadTuning {
    schema_version: u32,
    key: String,
    threads: RuntimeThreadCounts,
}

#[derive(Debug, Clone, Copy)]
struct ThreadBenchmarkSample {
    threads: u32,
    batch_elapsed: Duration,
    generation_elapsed: Duration,
}

#[derive(Debug, Default)]
struct ThreadBenchmarkMeasurements {
    batch: Vec<Duration>,
    generation: Vec<Duration>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuInstructionCapabilities {
    avx2: bool,
    fma: bool,
    f16c: bool,
    avx512f: bool,
    avx512cd: bool,
    avx512bw: bool,
    avx512dq: bool,
    avx512vl: bool,
}

impl ThreadTuningStore {
    pub(super) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            memory: Mutex::new(BTreeMap::new()),
            pending_next_launch: Mutex::new(BTreeSet::new()),
        }
    }

    #[cfg(feature = "embedded-llama")]
    pub(super) fn resolve(
        &self,
        model: &ModelDefinition,
        model_path: &Path,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeThreadCounts> {
        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return parse_runtime_threads(runtime);
        }

        let available_threads = available_runtime_threads();
        let fallback = automatic_runtime_thread_counts(available_threads);
        let key = thread_tuning_key(model, model_path, runtime, available_threads)?;
        if self.is_pending_next_launch(&key)? {
            trace_thread_tuning("pending_next_launch", 1);
            return Ok(fallback);
        }
        trace_thread_tuning("pending_next_launch", 0);
        if let Some(stored) = self.load(&key, available_threads)? {
            trace_thread_tuning("cache_hit", 1);
            return Ok(stored);
        }
        trace_thread_tuning("cache_hit", 0);
        Ok(fallback)
    }

    #[cfg(feature = "embedded-llama")]
    pub(super) fn tune(
        &self,
        llama_model: &llama_cpp::LlamaModel,
        model: &ModelDefinition,
        model_path: &Path,
        runtime: &RuntimeDefinition,
        is_cancelled: impl Fn() -> bool,
    ) -> Result<bool> {
        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return Ok(false);
        }

        let available_threads = available_runtime_threads();
        let key = thread_tuning_key(model, model_path, runtime, available_threads)?;
        if self.is_pending_next_launch(&key)? {
            trace_thread_tuning("tune_skipped", 1);
            return Ok(false);
        }
        if self.load(&key, available_threads)?.is_some() {
            trace_thread_tuning("tune_skipped", 1);
            return Ok(false);
        }
        trace_thread_tuning("tune_skipped", 0);

        let tuned = match benchmark_runtime_threads(llama_model, available_threads, is_cancelled) {
            Ok(Some(tuned)) => tuned,
            Ok(None) => {
                trace_thread_tuning("cancelled", 1);
                return Ok(false);
            }
            Err(error) => {
                eprintln!("embedded thread tuning failed: {error}");
                return Ok(false);
            }
        };
        if let Err(error) = self.store(&key, tuned) {
            eprintln!("failed to persist embedded thread tuning: {error}");
        }
        Ok(true)
    }

    pub(super) fn reset(
        &self,
        model: &ModelDefinition,
        model_path: &Path,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return Ok(false);
        }

        let key = thread_tuning_key(model, model_path, runtime, available_runtime_threads())?;
        self.memory
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning cache is unavailable".into()))?
            .remove(&key);
        self.pending_next_launch
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning state is unavailable".into()))?
            .remove(&key);
        let path = self.record_path(&key);
        match fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    pub(super) fn is_required(
        &self,
        model: &ModelDefinition,
        model_path: &Path,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return Ok(false);
        }

        let available_threads = available_runtime_threads();
        let key = thread_tuning_key(model, model_path, runtime, available_threads)?;
        Ok(self.load(&key, available_threads)?.is_none())
    }

    fn load(&self, key: &str, available_threads: usize) -> Result<Option<RuntimeThreadCounts>> {
        if let Some(stored) = self
            .memory
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning cache is unavailable".into()))?
            .get(key)
            .copied()
        {
            return Ok(Some(stored));
        }

        let path = self.record_path(key);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let record: PersistedThreadTuning = match serde_json::from_slice(&bytes) {
            Ok(record) => record,
            Err(_) => {
                let _ = fs::remove_file(path);
                return Ok(None);
            }
        };
        if record.schema_version != THREAD_TUNING_SCHEMA_VERSION
            || record.key != key
            || !valid_thread_counts(record.threads, available_threads)
        {
            let _ = fs::remove_file(path);
            return Ok(None);
        }

        self.memory
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning cache is unavailable".into()))?
            .insert(key.to_string(), record.threads);
        Ok(Some(record.threads))
    }

    fn store(&self, key: &str, threads: RuntimeThreadCounts) -> Result<()> {
        fs::create_dir_all(&self.directory)?;
        let target_path = self.record_path(key);
        if !target_path.is_file() {
            let record = PersistedThreadTuning {
                schema_version: THREAD_TUNING_SCHEMA_VERSION,
                key: key.to_string(),
                threads,
            };
            let bytes = serde_json::to_vec_pretty(&record).map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to serialize thread tuning record: {error}"
                ))
            })?;
            let temp_path = self.directory.join(format!(
                ".thread-tuning-{}-{}.tmp",
                std::process::id(),
                Uuid::new_v4()
            ));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);

            match fs::rename(&temp_path, &target_path) {
                Ok(()) => {}
                Err(_error) if target_path.is_file() => {
                    let _ = fs::remove_file(temp_path);
                }
                Err(error) => {
                    let _ = fs::remove_file(temp_path);
                    return Err(error.into());
                }
            }
        }

        self.pending_next_launch
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning state is unavailable".into()))?
            .insert(key.to_string());
        Ok(())
    }

    fn is_pending_next_launch(&self, key: &str) -> Result<bool> {
        Ok(self
            .pending_next_launch
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning state is unavailable".into()))?
            .contains(key))
    }

    fn record_path(&self, key: &str) -> PathBuf {
        self.directory
            .join(format!("{}.json", sha256_hex(key.as_bytes())))
    }
}

pub(super) fn automatic_runtime_thread_counts(available_threads: usize) -> RuntimeThreadCounts {
    let reserved_threads = usize::from(available_threads > 1);
    let batch = available_threads
        .saturating_sub(reserved_threads)
        .clamp(1, MAX_AUTOMATIC_RUNTIME_THREADS) as u32;
    let generation = if batch >= 7 { batch - 1 } else { batch };

    RuntimeThreadCounts { generation, batch }
}

pub(super) fn parse_runtime_threads(runtime: &RuntimeDefinition) -> Result<RuntimeThreadCounts> {
    if runtime.threads.eq_ignore_ascii_case("auto") {
        return Ok(automatic_runtime_thread_counts(available_runtime_threads()));
    }

    let threads = runtime.threads.parse::<u32>().map_err(|error| {
        CompressionError::InvalidConfig(format!(
            "runtime '{}' has invalid threads value '{}': {error}",
            runtime.id, runtime.threads
        ))
    })?;
    if threads == 0 {
        return Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' threads must be greater than zero",
            runtime.id
        )));
    }

    Ok(RuntimeThreadCounts {
        generation: threads,
        batch: threads,
    })
}

fn available_runtime_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn thread_tuning_candidates(available_threads: usize) -> Vec<u32> {
    let reserved_threads = usize::from(available_threads > 1);
    let maximum = available_threads
        .saturating_sub(reserved_threads)
        .clamp(1, MAX_AUTOMATIC_RUNTIME_THREADS);
    let minimum = maximum.saturating_sub(3).max(1);
    (minimum..=maximum).map(|value| value as u32).collect()
}

#[cfg(feature = "embedded-llama")]
fn benchmark_runtime_threads(
    llama_model: &llama_cpp::LlamaModel,
    available_threads: usize,
    is_cancelled: impl Fn() -> bool,
) -> Result<Option<RuntimeThreadCounts>> {
    let candidates = thread_tuning_candidates(available_threads);
    if candidates.len() == 1 {
        return Ok(Some(RuntimeThreadCounts {
            generation: candidates[0],
            batch: candidates[0],
        }));
    }

    let benchmark_text = "TrimPromptのCPU推論速度を測定する固定文章です。入力評価と逐次生成の処理時間だけを比較します。".repeat(20);
    let tokens = llama_model
        .tokenize_bytes(benchmark_text.as_bytes(), false, true)
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to tokenize thread benchmark: {error}"))
        })?;
    let required_tokens =
        BENCHMARK_WARMUP_TOKENS + BENCHMARK_BATCH_TOKENS + BENCHMARK_GENERATION_TOKENS;
    if tokens.len() < required_tokens {
        return Err(CompressionError::Runtime(
            "thread benchmark token sequence is too short".into(),
        ));
    }

    // 最初の候補だけがモデルページ読み込みの影響を受けないよう、共通のウォームアップを先に行う。
    let fallback = automatic_runtime_thread_counts(available_threads);
    let mut warmup = llama_model
        .create_session(benchmark_session_params(fallback))
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to create thread benchmark warmup: {error}"))
        })?;
    warmup
        .advance_context_with_tokens(&tokens[..BENCHMARK_WARMUP_TOKENS])
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to warm thread benchmark: {error}"))
        })?;
    drop(warmup);

    // 候補を交互に測り、温度や一時的なバックグラウンド負荷が一候補へ偏るのを防ぐ。
    let mut measurements = BTreeMap::<u32, ThreadBenchmarkMeasurements>::new();
    for _round in 0..BENCHMARK_INITIAL_ROUNDS {
        for &threads in &candidates {
            if is_cancelled() {
                return Ok(None);
            }
            let sample = run_thread_benchmark_sample(llama_model, &tokens, threads)?;
            let entry = measurements.entry(threads).or_default();
            entry.batch.push(sample.batch_elapsed);
            entry.generation.push(sample.generation_elapsed);
            trace_thread_tuning_duration(threads, "batch_sample", sample.batch_elapsed);
            trace_thread_tuning_duration(threads, "generation_sample", sample.generation_elapsed);
        }
    }

    let unstable_candidates = measurements
        .iter()
        .filter(|(_, values)| {
            measurements_are_unstable(&values.batch)
                || measurements_are_unstable(&values.generation)
        })
        .map(|(&threads, _)| threads)
        .collect::<Vec<_>>();
    for _round in 0..BENCHMARK_UNSTABLE_EXTRA_ROUNDS {
        for &threads in &unstable_candidates {
            if is_cancelled() {
                return Ok(None);
            }
            let sample = run_thread_benchmark_sample(llama_model, &tokens, threads)?;
            let entry = measurements
                .get_mut(&threads)
                .expect("unstable candidate must have initial measurements");
            entry.batch.push(sample.batch_elapsed);
            entry.generation.push(sample.generation_elapsed);
        }
    }

    let samples = candidates
        .iter()
        .map(|&threads| {
            let values = measurements
                .get(&threads)
                .expect("each candidate must have measurements");
            ThreadBenchmarkSample {
                threads,
                batch_elapsed: median_duration(&values.batch),
                generation_elapsed: median_duration(&values.generation),
            }
        })
        .collect::<Vec<_>>();
    for sample in &samples {
        trace_thread_tuning_duration(sample.threads, "batch", sample.batch_elapsed);
        trace_thread_tuning_duration(sample.threads, "generation", sample.generation_elapsed);
    }

    Ok(Some(RuntimeThreadCounts {
        generation: select_preferred_thread_count(&samples, |sample| sample.generation_elapsed),
        batch: select_preferred_thread_count(&samples, |sample| sample.batch_elapsed),
    }))
}

#[cfg(feature = "embedded-llama")]
fn run_thread_benchmark_sample(
    llama_model: &llama_cpp::LlamaModel,
    tokens: &[llama_cpp::Token],
    threads: u32,
) -> Result<ThreadBenchmarkSample> {
    let counts = RuntimeThreadCounts {
        generation: threads,
        batch: threads,
    };
    let mut session = llama_model
        .create_session(benchmark_session_params(counts))
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to create {threads}-thread benchmark session: {error}"
            ))
        })?;
    session
        .advance_context_with_tokens(&tokens[..BENCHMARK_WARMUP_TOKENS])
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to warm {threads}-thread benchmark session: {error}"
            ))
        })?;

    let batch_end = BENCHMARK_WARMUP_TOKENS + BENCHMARK_BATCH_TOKENS;
    let required_tokens = batch_end + BENCHMARK_GENERATION_TOKENS;
    let batch_started_at = std::time::Instant::now();
    session
        .advance_context_with_tokens(&tokens[BENCHMARK_WARMUP_TOKENS..batch_end])
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to run {threads}-thread batch benchmark: {error}"
            ))
        })?;
    let batch_elapsed = batch_started_at.elapsed();

    let generation_started_at = std::time::Instant::now();
    for token in &tokens[batch_end..required_tokens] {
        session
            .advance_context_with_tokens(std::slice::from_ref(token))
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to run {threads}-thread generation benchmark: {error}"
                ))
            })?;
    }

    Ok(ThreadBenchmarkSample {
        threads,
        batch_elapsed,
        generation_elapsed: generation_started_at.elapsed(),
    })
}

fn median_duration(values: &[Duration]) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted
        .get(sorted.len() / 2)
        .copied()
        .unwrap_or(Duration::ZERO)
}

fn measurements_are_unstable(values: &[Duration]) -> bool {
    let Some(minimum) = values.iter().map(Duration::as_nanos).min() else {
        return false;
    };
    let Some(maximum) = values.iter().map(Duration::as_nanos).max() else {
        return false;
    };
    minimum > 0 && maximum * 100 > minimum * BENCHMARK_STABLE_PERCENT
}

#[cfg(feature = "embedded-llama")]
fn benchmark_session_params(threads: RuntimeThreadCounts) -> llama_cpp::SessionParams {
    llama_cpp::SessionParams {
        n_ctx: BENCHMARK_CONTEXT_LENGTH,
        n_threads: threads.generation,
        n_threads_batch: threads.batch,
    }
}

fn select_preferred_thread_count(
    samples: &[ThreadBenchmarkSample],
    elapsed: impl Fn(&ThreadBenchmarkSample) -> Duration,
) -> u32 {
    let fastest = samples
        .iter()
        .map(|sample| elapsed(sample).as_nanos())
        .min()
        .unwrap_or(1);
    samples
        .iter()
        .filter(|sample| elapsed(sample).as_nanos() * 100 <= fastest * NEAR_FASTEST_PERCENT)
        .map(|sample| sample.threads)
        .min()
        .or_else(|| {
            samples
                .iter()
                .min_by_key(|sample| elapsed(sample))
                .map(|sample| sample.threads)
        })
        .unwrap_or(1)
}

fn valid_thread_counts(threads: RuntimeThreadCounts, available_threads: usize) -> bool {
    let maximum = automatic_runtime_thread_counts(available_threads).batch;
    (1..=maximum).contains(&threads.generation) && (1..=maximum).contains(&threads.batch)
}

fn thread_tuning_key(
    model: &ModelDefinition,
    model_path: &Path,
    runtime: &RuntimeDefinition,
    available_threads: usize,
) -> Result<String> {
    let metadata = fs::metadata(model_path)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let processor = env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string());
    let instruction_profile = detected_cpu_instruction_profile();
    Ok(format!(
        "contract={THREAD_TUNING_CONTRACT_VERSION}|cpu={processor}|instructions={instruction_profile}|engine={}|available={available_threads}|model={}|size={}|modified={modified}|runtime={}|threads={}",
        compiled_cpu_engine(),
        model.id,
        metadata.len(),
        runtime.id,
        runtime.threads
    ))
}

fn compiled_cpu_engine() -> &'static str {
    if cfg!(feature = "embedded-llama-avx512") {
        "avx512"
    } else if cfg!(feature = "embedded-llama-avx2") {
        "avx2"
    } else {
        "compatible"
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detected_cpu_instruction_profile() -> &'static str {
    cpu_instruction_profile(CpuInstructionCapabilities {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512cd: std::arch::is_x86_feature_detected!("avx512cd"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512dq: std::arch::is_x86_feature_detected!("avx512dq"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detected_cpu_instruction_profile() -> &'static str {
    "compatible"
}

fn cpu_instruction_profile(capabilities: CpuInstructionCapabilities) -> &'static str {
    if capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.avx512f
        && capabilities.avx512cd
        && capabilities.avx512bw
        && capabilities.avx512dq
        && capabilities.avx512vl
    {
        "avx512-core"
    } else if capabilities.avx2 && capabilities.fma && capabilities.f16c {
        "avx2-fma-f16c"
    } else {
        "compatible"
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn trace_thread_tuning(stage: &str, value: usize) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE").is_some() {
        eprintln!("trace.runtime.thread_tuning.{stage}={value}");
    }
}

fn trace_thread_tuning_duration(threads: u32, stage: &str, elapsed: Duration) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE").is_some() {
        eprintln!(
            "trace.runtime.thread_tuning.{threads}.{stage}_ms={}",
            elapsed.as_millis()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_candidates_cover_at_most_the_four_highest_supported_counts() {
        assert_eq!(thread_tuning_candidates(1), [1]);
        assert_eq!(thread_tuning_candidates(4), [1, 2, 3]);
        assert_eq!(thread_tuning_candidates(8), [4, 5, 6, 7]);
        assert_eq!(thread_tuning_candidates(16), [5, 6, 7, 8]);
    }

    #[test]
    fn near_fastest_selection_prefers_fewer_threads() {
        let samples = [
            ThreadBenchmarkSample {
                threads: 5,
                batch_elapsed: Duration::from_millis(103),
                generation_elapsed: Duration::from_millis(110),
            },
            ThreadBenchmarkSample {
                threads: 6,
                batch_elapsed: Duration::from_millis(100),
                generation_elapsed: Duration::from_millis(100),
            },
            ThreadBenchmarkSample {
                threads: 7,
                batch_elapsed: Duration::from_millis(100),
                generation_elapsed: Duration::from_millis(99),
            },
        ];

        assert_eq!(
            select_preferred_thread_count(&samples, |sample| sample.batch_elapsed),
            5
        );
        assert_eq!(
            select_preferred_thread_count(&samples, |sample| sample.generation_elapsed),
            6
        );
    }

    #[test]
    fn tuning_record_round_trips_and_rejects_a_different_key() {
        let directory =
            std::env::temp_dir().join(format!("trim-prompt-thread-tuning-{}", Uuid::new_v4()));
        let store = ThreadTuningStore::new(directory.clone());
        let threads = RuntimeThreadCounts {
            generation: 6,
            batch: 7,
        };

        store.store("machine-a", threads).expect("store tuning");
        assert!(store
            .is_pending_next_launch("machine-a")
            .expect("read pending tuning state"));
        assert_eq!(
            store.load("machine-a", 8).expect("load tuning"),
            Some(threads)
        );
        assert_eq!(store.load("machine-b", 8).expect("miss tuning"), None);

        let fresh_store = ThreadTuningStore::new(directory.clone());
        assert!(!fresh_store
            .is_pending_next_launch("machine-a")
            .expect("read fresh tuning state"));
        assert_eq!(
            fresh_store
                .load("machine-a", 8)
                .expect("load after restart"),
            Some(threads)
        );

        fs::remove_dir_all(directory).expect("remove tuning test directory");
    }

    #[test]
    fn instruction_profile_requires_all_avx2_engine_features() {
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities {
                avx2: true,
                fma: true,
                f16c: true,
                ..Default::default()
            }),
            "avx2-fma-f16c"
        );
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities {
                avx2: true,
                fma: true,
                f16c: true,
                avx512f: true,
                avx512cd: true,
                avx512bw: true,
                avx512dq: true,
                avx512vl: true,
            }),
            "avx512-core"
        );
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities {
                avx2: true,
                f16c: true,
                ..Default::default()
            }),
            "compatible"
        );
    }

    #[test]
    fn median_and_stability_ignore_one_slow_measurement() {
        let values = [
            Duration::from_millis(100),
            Duration::from_millis(101),
            Duration::from_millis(180),
        ];
        assert_eq!(median_duration(&values), Duration::from_millis(101));
        assert!(measurements_are_unstable(&values));
        assert!(!measurements_are_unstable(&[
            Duration::from_millis(100),
            Duration::from_millis(105),
            Duration::from_millis(110),
        ]));
    }
}

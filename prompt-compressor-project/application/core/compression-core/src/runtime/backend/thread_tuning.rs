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

const THREAD_TUNING_SCHEMA_VERSION: u32 = 2;
const THREAD_TUNING_CONTRACT_VERSION: u32 = 11;
const MAX_AUTOMATIC_RUNTIME_THREADS: usize = 8;
const THREAD_MODE_ENV: &str = "TRIMPROMPT_THREAD_MODE";
const GENERATION_THREADS_ENV: &str = "TRIMPROMPT_GENERATION_THREADS";
const BATCH_THREADS_ENV: &str = "TRIMPROMPT_BATCH_THREADS";
const BENCHMARK_CONTEXT_LENGTH: u32 = 1_024;
const BENCHMARK_WARMUP_TOKENS: usize = 16;
const BENCHMARK_BATCH_TOKENS: usize = 128;
const BENCHMARK_GENERATION_TOKENS: usize = 12;
const BATCH_SIZE_BENCHMARK_TOKENS: usize = 512;
const BATCH_SIZE_QUALITY_CONTEXT_LENGTH: u32 = 2_048;
const BATCH_SIZE_QUALITY_MAX_PREDICTIONS: usize = 192;
const BENCHMARK_INITIAL_ROUNDS: usize = 3;
const BENCHMARK_UNSTABLE_EXTRA_ROUNDS: usize = 2;
const BENCHMARK_STABLE_PERCENT: u128 = 120;
const NEAR_FASTEST_PERCENT: u128 = 103;
const DEFAULT_LOGICAL_BATCH_SIZE: u32 = 512;
const DEFAULT_PHYSICAL_BATCH_SIZE: u32 = 512;
const PHYSICAL_BATCH_SIZE_CANDIDATES: [u32; 3] = [128, 256, 512];
const PHYSICAL_BATCH_MINIMUM_GAIN_PERCENT: u128 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct RuntimeThreadCounts {
    pub(super) generation: u32,
    pub(super) batch: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct RuntimeBatchSizes {
    pub(super) logical: u32,
    pub(super) physical: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct RuntimeInferenceConfig {
    pub(super) threads: RuntimeThreadCounts,
    pub(super) batch_sizes: RuntimeBatchSizes,
}

#[derive(Debug)]
pub(super) struct ThreadTuningStore {
    directory: PathBuf,
    memory: Mutex<BTreeMap<String, RuntimeInferenceConfig>>,
    pending_next_launch: Mutex<BTreeSet<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedThreadTuning {
    schema_version: u32,
    key: String,
    configuration: RuntimeInferenceConfig,
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

#[derive(Debug, Clone, Copy)]
struct PhysicalBatchBenchmarkSample {
    physical_batch_size: u32,
    elapsed: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuInstructionCapabilities {
    sse42: bool,
    avx2: bool,
    fma: bool,
    f16c: bool,
    bmi2: bool,
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
    ) -> Result<RuntimeInferenceConfig> {
        if let Some(threads) = manual_runtime_thread_counts()? {
            trace_thread_tuning("manual_override", 1);
            return Ok(runtime_config_for_threads(threads));
        }
        trace_thread_tuning("manual_override", 0);

        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return parse_runtime_threads(runtime).map(runtime_config_for_threads);
        }

        let available_threads = available_runtime_threads();
        let fallback =
            runtime_config_for_threads(automatic_runtime_thread_counts(available_threads));
        let key = thread_tuning_key(model, model_path, runtime, available_threads)?;
        if self.is_pending_next_launch(&key)? {
            trace_thread_tuning("pending_next_launch", 1);
            if let Some(stored) = self.load(&key, available_threads)? {
                trace_thread_tuning("cache_hit", 1);
                return Ok(stored);
            }
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
        quality_prompts: &[Vec<u8>],
        is_cancelled: impl Fn() -> bool,
    ) -> Result<bool> {
        if manual_runtime_thread_counts()?.is_some() {
            return Ok(false);
        }

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

        let tuned = match benchmark_runtime_configuration(
            llama_model,
            available_threads,
            quality_prompts,
            &is_cancelled,
        ) {
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
        self.store(&key, tuned)?;
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
        if manual_runtime_thread_counts()?.is_some() {
            return Ok(false);
        }

        if !runtime.threads.eq_ignore_ascii_case("auto") {
            return Ok(false);
        }

        let available_threads = available_runtime_threads();
        let key = thread_tuning_key(model, model_path, runtime, available_threads)?;
        Ok(self.load(&key, available_threads)?.is_none())
    }

    fn load(&self, key: &str, available_threads: usize) -> Result<Option<RuntimeInferenceConfig>> {
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
            || !valid_runtime_configuration(record.configuration, available_threads)
        {
            let _ = fs::remove_file(path);
            return Ok(None);
        }

        self.memory
            .lock()
            .map_err(|_| CompressionError::Runtime("thread tuning cache is unavailable".into()))?
            .insert(key.to_string(), record.configuration);
        Ok(Some(record.configuration))
    }

    fn store(&self, key: &str, configuration: RuntimeInferenceConfig) -> Result<()> {
        fs::create_dir_all(&self.directory)?;
        let target_path = self.record_path(key);
        if !target_path.is_file() {
            let record = PersistedThreadTuning {
                schema_version: THREAD_TUNING_SCHEMA_VERSION,
                key: key.to_string(),
                configuration,
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

fn default_runtime_batch_sizes() -> RuntimeBatchSizes {
    RuntimeBatchSizes {
        logical: DEFAULT_LOGICAL_BATCH_SIZE,
        physical: DEFAULT_PHYSICAL_BATCH_SIZE,
    }
}

pub(super) fn runtime_config_for_threads(threads: RuntimeThreadCounts) -> RuntimeInferenceConfig {
    RuntimeInferenceConfig {
        threads,
        batch_sizes: default_runtime_batch_sizes(),
    }
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

pub(super) fn manual_runtime_thread_counts() -> Result<Option<RuntimeThreadCounts>> {
    let mode = env::var(THREAD_MODE_ENV).ok();
    let generation = env::var(GENERATION_THREADS_ENV).ok();
    let batch = env::var(BATCH_THREADS_ENV).ok();
    manual_runtime_thread_counts_from_values(
        mode.as_deref(),
        generation.as_deref(),
        batch.as_deref(),
        available_runtime_threads(),
    )
}

fn manual_runtime_thread_counts_from_values(
    mode: Option<&str>,
    generation: Option<&str>,
    batch: Option<&str>,
    available_threads: usize,
) -> Result<Option<RuntimeThreadCounts>> {
    if mode != Some("manual") {
        return Ok(None);
    }

    let parse = |name: &str, value: Option<&str>| -> Result<u32> {
        let value = value.ok_or_else(|| {
            CompressionError::InvalidConfig(format!("manual {name} thread count is missing"))
        })?;
        let threads = value.parse::<u32>().map_err(|error| {
            CompressionError::InvalidConfig(format!(
                "manual {name} thread count '{value}' is invalid: {error}"
            ))
        })?;
        let maximum = u32::try_from(available_threads).unwrap_or(u32::MAX).max(1);
        if !(1..=maximum).contains(&threads) {
            return Err(CompressionError::InvalidConfig(format!(
                "manual {name} thread count must be between 1 and {maximum}"
            )));
        }
        Ok(threads)
    };

    Ok(Some(RuntimeThreadCounts {
        generation: parse("generation", generation)?,
        batch: parse("batch", batch)?,
    }))
}

pub(super) fn available_runtime_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn thread_tuning_candidates(available_threads: usize) -> Vec<u32> {
    let reserved_threads = usize::from(available_threads > 1);
    let maximum = available_threads
        .saturating_sub(reserved_threads)
        .clamp(1, MAX_AUTOMATIC_RUNTIME_THREADS);
    if maximum <= 4 {
        return (1..=maximum).map(|value| value as u32).collect();
    }

    // 高スレッド候補に加え、帯域制約の強いCPUで有利になりやすい2・4も必ず実測する。
    let minimum = maximum.saturating_sub(3).max(1);
    [2, 4]
        .into_iter()
        .chain(minimum..=maximum)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|value| value as u32)
        .collect()
}

#[cfg(feature = "embedded-llama")]
fn benchmark_runtime_configuration(
    llama_model: &llama_cpp::LlamaModel,
    available_threads: usize,
    quality_prompts: &[Vec<u8>],
    is_cancelled: &impl Fn() -> bool,
) -> Result<Option<RuntimeInferenceConfig>> {
    let Some(threads) = benchmark_runtime_threads(llama_model, available_threads, is_cancelled)?
    else {
        return Ok(None);
    };
    let physical_batch_size =
        match benchmark_physical_batch_size(llama_model, threads, quality_prompts, is_cancelled) {
            Ok(Some(physical_batch_size)) => physical_batch_size,
            Ok(None) => return Ok(None),
            Err(error) => {
                eprintln!(
                    "physical batch tuning failed; keeping {DEFAULT_PHYSICAL_BATCH_SIZE}: {error}"
                );
                DEFAULT_PHYSICAL_BATCH_SIZE
            }
        };
    Ok(Some(RuntimeInferenceConfig {
        threads,
        batch_sizes: RuntimeBatchSizes {
            logical: DEFAULT_LOGICAL_BATCH_SIZE,
            physical: physical_batch_size,
        },
    }))
}

#[cfg(feature = "embedded-llama")]
fn benchmark_runtime_threads(
    llama_model: &llama_cpp::LlamaModel,
    available_threads: usize,
    is_cancelled: &impl Fn() -> bool,
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

#[cfg(feature = "embedded-llama")]
fn benchmark_physical_batch_size(
    llama_model: &llama_cpp::LlamaModel,
    threads: RuntimeThreadCounts,
    quality_prompts: &[Vec<u8>],
    is_cancelled: &impl Fn() -> bool,
) -> Result<Option<u32>> {
    let benchmark_text = "TrimPromptの長文入力評価に適したマイクロバッチサイズを測定する固定文章です。モデルや生成条件は変更せず、CPUの計算単位だけを比較します。".repeat(80);
    let tokens = llama_model
        .tokenize_bytes(benchmark_text.as_bytes(), false, true)
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to tokenize physical batch benchmark: {error}"
            ))
        })?;
    let required_tokens = BENCHMARK_WARMUP_TOKENS + BATCH_SIZE_BENCHMARK_TOKENS;
    if tokens.len() < required_tokens {
        return Err(CompressionError::Runtime(
            "physical batch benchmark token sequence is too short".into(),
        ));
    }

    // 比較開始前に既定値でモデルページを読み込み、最初の候補だけが不利になるのを防ぐ。
    let mut warmup = llama_model
        .create_session(benchmark_session_params_with_batch_size(
            threads,
            DEFAULT_PHYSICAL_BATCH_SIZE,
        ))
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to create physical batch benchmark warmup: {error}"
            ))
        })?;
    warmup
        .advance_context_with_tokens(&tokens[..BENCHMARK_WARMUP_TOKENS])
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to warm physical batch benchmark: {error}"))
        })?;
    drop(warmup);

    let mut measurements = BTreeMap::<u32, Vec<Duration>>::new();
    for round in 0..BENCHMARK_INITIAL_ROUNDS {
        // ラウンドごとに開始候補をずらし、温度変化やバックグラウンド負荷の偏りを抑える。
        for offset in 0..PHYSICAL_BATCH_SIZE_CANDIDATES.len() {
            if is_cancelled() {
                return Ok(None);
            }
            let index = (round + offset) % PHYSICAL_BATCH_SIZE_CANDIDATES.len();
            let physical_batch_size = PHYSICAL_BATCH_SIZE_CANDIDATES[index];
            let elapsed = run_physical_batch_benchmark_sample(
                llama_model,
                &tokens,
                threads,
                physical_batch_size,
            )?;
            measurements
                .entry(physical_batch_size)
                .or_default()
                .push(elapsed);
            trace_physical_batch_tuning_duration(physical_batch_size, "sample", elapsed);
        }
    }

    let unstable_candidates = measurements
        .iter()
        .filter(|(_, values)| measurements_are_unstable(values))
        .map(|(&physical_batch_size, _)| physical_batch_size)
        .collect::<Vec<_>>();
    for _round in 0..BENCHMARK_UNSTABLE_EXTRA_ROUNDS {
        for &physical_batch_size in &unstable_candidates {
            if is_cancelled() {
                return Ok(None);
            }
            let elapsed = run_physical_batch_benchmark_sample(
                llama_model,
                &tokens,
                threads,
                physical_batch_size,
            )?;
            measurements
                .get_mut(&physical_batch_size)
                .expect("unstable physical batch candidate must have initial measurements")
                .push(elapsed);
        }
    }

    let samples = PHYSICAL_BATCH_SIZE_CANDIDATES.map(|physical_batch_size| {
        let elapsed = median_duration(
            measurements
                .get(&physical_batch_size)
                .expect("each physical batch candidate must have measurements"),
        );
        trace_physical_batch_tuning_duration(physical_batch_size, "median", elapsed);
        PhysicalBatchBenchmarkSample {
            physical_batch_size,
            elapsed,
        }
    });
    let speed_candidate = select_physical_batch_size(&samples);
    if speed_candidate == DEFAULT_PHYSICAL_BATCH_SIZE {
        return Ok(Some(DEFAULT_PHYSICAL_BATCH_SIZE));
    }

    let Some(quality_matches) = physical_batch_preserves_greedy_outputs(
        llama_model,
        threads,
        speed_candidate,
        quality_prompts,
        is_cancelled,
    )?
    else {
        return Ok(None);
    };
    trace_physical_batch_tuning_value(
        speed_candidate,
        "quality_match",
        usize::from(quality_matches),
    );
    Ok(Some(select_quality_preserving_physical_batch_size(
        speed_candidate,
        quality_matches,
    )))
}

#[cfg(feature = "embedded-llama")]
fn physical_batch_preserves_greedy_outputs(
    llama_model: &llama_cpp::LlamaModel,
    threads: RuntimeThreadCounts,
    physical_batch_size: u32,
    quality_prompts: &[Vec<u8>],
    is_cancelled: &impl Fn() -> bool,
) -> Result<Option<bool>> {
    if quality_prompts.is_empty() {
        return Ok(Some(false));
    }

    for (index, prompt) in quality_prompts.iter().enumerate() {
        if is_cancelled() {
            return Ok(None);
        }
        let baseline = generate_batch_quality_output(
            llama_model,
            threads,
            DEFAULT_PHYSICAL_BATCH_SIZE,
            prompt,
        )?;
        if is_cancelled() {
            return Ok(None);
        }
        let candidate =
            generate_batch_quality_output(llama_model, threads, physical_batch_size, prompt)?;
        let matches = baseline == candidate;
        trace_physical_batch_tuning_value(
            physical_batch_size,
            &format!("quality_prompt_{index}"),
            usize::from(matches),
        );
        if !matches {
            return Ok(Some(false));
        }
    }

    Ok(Some(true))
}

#[cfg(feature = "embedded-llama")]
fn generate_batch_quality_output(
    llama_model: &llama_cpp::LlamaModel,
    threads: RuntimeThreadCounts,
    physical_batch_size: u32,
    prompt: &[u8],
) -> Result<String> {
    let tokens = llama_model
        .tokenize_bytes(prompt, false, true)
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to tokenize {physical_batch_size}-token batch quality prompt: {error}"
            ))
        })?;
    if tokens.len() + BATCH_SIZE_QUALITY_MAX_PREDICTIONS
        > BATCH_SIZE_QUALITY_CONTEXT_LENGTH as usize
    {
        return Err(CompressionError::Runtime(format!(
            "{physical_batch_size}-token batch quality prompt exceeds its context"
        )));
    }
    let mut params = benchmark_session_params_with_batch_size(threads, physical_batch_size);
    params.n_ctx = BATCH_SIZE_QUALITY_CONTEXT_LENGTH;
    let mut session = llama_model.create_session(params).map_err(|error| {
        CompressionError::Runtime(format!(
            "failed to create {physical_batch_size}-token batch quality session: {error}"
        ))
    })?;
    session
        .advance_context_with_tokens(&tokens)
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to evaluate {physical_batch_size}-token batch quality prompt: {error}"
            ))
        })?;
    let output = session
        .start_completing_with(
            llama_cpp::standard_sampler::StandardSampler::new_greedy(),
            BATCH_SIZE_QUALITY_MAX_PREDICTIONS,
        )
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to start {physical_batch_size}-token batch quality completion: {error}"
            ))
        })?
        .into_strings()
        .collect::<std::result::Result<String, _>>()
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to generate {physical_batch_size}-token batch quality output: {error}"
            ))
        })?;
    Ok(output)
}

#[cfg(feature = "embedded-llama")]
fn run_physical_batch_benchmark_sample(
    llama_model: &llama_cpp::LlamaModel,
    tokens: &[llama_cpp::Token],
    threads: RuntimeThreadCounts,
    physical_batch_size: u32,
) -> Result<Duration> {
    let mut session = llama_model
        .create_session(benchmark_session_params_with_batch_size(
            threads,
            physical_batch_size,
        ))
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to create {physical_batch_size}-token physical batch benchmark: {error}"
            ))
        })?;
    session
        .advance_context_with_tokens(&tokens[..BENCHMARK_WARMUP_TOKENS])
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to warm {physical_batch_size}-token physical batch benchmark: {error}"
            ))
        })?;

    let batch_end = BENCHMARK_WARMUP_TOKENS + BATCH_SIZE_BENCHMARK_TOKENS;
    let started_at = std::time::Instant::now();
    session
        .advance_context_with_tokens(&tokens[BENCHMARK_WARMUP_TOKENS..batch_end])
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to run {physical_batch_size}-token physical batch benchmark: {error}"
            ))
        })?;
    Ok(started_at.elapsed())
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
    benchmark_session_params_with_batch_size(threads, DEFAULT_PHYSICAL_BATCH_SIZE)
}

#[cfg(feature = "embedded-llama")]
fn benchmark_session_params_with_batch_size(
    threads: RuntimeThreadCounts,
    physical_batch_size: u32,
) -> llama_cpp::SessionParams {
    llama_cpp::SessionParams {
        n_ctx: BENCHMARK_CONTEXT_LENGTH,
        n_batch: DEFAULT_LOGICAL_BATCH_SIZE,
        n_ubatch: physical_batch_size,
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

fn select_physical_batch_size(samples: &[PhysicalBatchBenchmarkSample]) -> u32 {
    let Some(default_elapsed) = samples
        .iter()
        .find(|sample| sample.physical_batch_size == DEFAULT_PHYSICAL_BATCH_SIZE)
        .map(|sample| sample.elapsed.as_nanos())
    else {
        return DEFAULT_PHYSICAL_BATCH_SIZE;
    };
    let Some(fastest) = samples.iter().min_by_key(|sample| sample.elapsed) else {
        return DEFAULT_PHYSICAL_BATCH_SIZE;
    };
    let required = 100 - PHYSICAL_BATCH_MINIMUM_GAIN_PERCENT;
    if fastest.physical_batch_size != DEFAULT_PHYSICAL_BATCH_SIZE
        && fastest.elapsed.as_nanos() * 100 <= default_elapsed * required
    {
        fastest.physical_batch_size
    } else {
        DEFAULT_PHYSICAL_BATCH_SIZE
    }
}

fn select_quality_preserving_physical_batch_size(
    speed_candidate: u32,
    quality_matches: bool,
) -> u32 {
    if speed_candidate != DEFAULT_PHYSICAL_BATCH_SIZE && quality_matches {
        speed_candidate
    } else {
        DEFAULT_PHYSICAL_BATCH_SIZE
    }
}

fn valid_runtime_configuration(
    configuration: RuntimeInferenceConfig,
    available_threads: usize,
) -> bool {
    let threads = configuration.threads;
    let maximum = automatic_runtime_thread_counts(available_threads).batch;
    (1..=maximum).contains(&threads.generation)
        && (1..=maximum).contains(&threads.batch)
        && configuration.batch_sizes.logical == DEFAULT_LOGICAL_BATCH_SIZE
        && PHYSICAL_BATCH_SIZE_CANDIDATES.contains(&configuration.batch_sizes.physical)
        && configuration.batch_sizes.physical <= configuration.batch_sizes.logical
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
        sse42: std::arch::is_x86_feature_detected!("sse4.2"),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
        bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512cd: std::arch::is_x86_feature_detected!("avx512cd"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512dq: std::arch::is_x86_feature_detected!("avx512dq"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detected_cpu_instruction_profile() -> &'static str {
    "unsupported"
}

fn cpu_instruction_profile(capabilities: CpuInstructionCapabilities) -> &'static str {
    if capabilities.sse42
        && capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.bmi2
        && capabilities.avx512f
        && capabilities.avx512cd
        && capabilities.avx512bw
        && capabilities.avx512dq
        && capabilities.avx512vl
    {
        "avx512-core"
    } else if capabilities.sse42
        && capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.bmi2
    {
        "avx2-fma-f16c-bmi2"
    } else if capabilities.sse42 {
        "sse4.2"
    } else {
        "unsupported"
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

fn trace_physical_batch_tuning_duration(physical_batch_size: u32, stage: &str, elapsed: Duration) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE").is_some() {
        eprintln!(
            "trace.runtime.batch_tuning.{physical_batch_size}.{stage}_ms={}",
            elapsed.as_millis()
        );
    }
}

fn trace_physical_batch_tuning_value(physical_batch_size: u32, stage: &str, value: usize) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE").is_some() {
        eprintln!("trace.runtime.batch_tuning.{physical_batch_size}.{stage}={value}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_candidates_cover_low_bandwidth_and_high_thread_counts() {
        assert_eq!(thread_tuning_candidates(1), [1]);
        assert_eq!(thread_tuning_candidates(4), [1, 2, 3]);
        assert_eq!(thread_tuning_candidates(8), [2, 4, 5, 6, 7]);
        assert_eq!(thread_tuning_candidates(16), [2, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn automatic_tuning_keeps_a_representative_benchmark_workload() {
        assert_eq!(BENCHMARK_WARMUP_TOKENS, 16);
        assert_eq!(BENCHMARK_BATCH_TOKENS, 128);
        assert_eq!(BENCHMARK_GENERATION_TOKENS, 12);
        assert_eq!(BATCH_SIZE_BENCHMARK_TOKENS, 512);
        assert_eq!(BATCH_SIZE_QUALITY_CONTEXT_LENGTH, 2_048);
        assert_eq!(BATCH_SIZE_QUALITY_MAX_PREDICTIONS, 192);
        assert_eq!(BENCHMARK_INITIAL_ROUNDS, 3);
        assert_eq!(BENCHMARK_UNSTABLE_EXTRA_ROUNDS, 2);
        assert_eq!(PHYSICAL_BATCH_SIZE_CANDIDATES, [128, 256, 512]);
    }

    #[test]
    fn manual_thread_counts_require_complete_values_within_the_cpu_limit() {
        assert_eq!(
            manual_runtime_thread_counts_from_values(Some("manual"), Some("3"), Some("4"), 4)
                .expect("valid manual values"),
            Some(RuntimeThreadCounts {
                generation: 3,
                batch: 4,
            })
        );
        assert!(
            manual_runtime_thread_counts_from_values(Some("manual"), Some("5"), Some("4"), 4)
                .is_err()
        );
        assert!(
            manual_runtime_thread_counts_from_values(Some("manual"), Some("3"), None, 4).is_err()
        );
        assert_eq!(
            manual_runtime_thread_counts_from_values(Some("auto"), None, None, 4)
                .expect("automatic mode"),
            None
        );
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
    fn physical_batch_selection_requires_a_measurable_gain() {
        let default = PhysicalBatchBenchmarkSample {
            physical_batch_size: 512,
            elapsed: Duration::from_millis(100),
        };
        assert_eq!(
            select_physical_batch_size(&[
                PhysicalBatchBenchmarkSample {
                    physical_batch_size: 128,
                    elapsed: Duration::from_millis(97),
                },
                PhysicalBatchBenchmarkSample {
                    physical_batch_size: 256,
                    elapsed: Duration::from_millis(99),
                },
                default,
            ]),
            128
        );
        assert_eq!(
            select_physical_batch_size(&[
                PhysicalBatchBenchmarkSample {
                    physical_batch_size: 128,
                    elapsed: Duration::from_millis(98),
                },
                PhysicalBatchBenchmarkSample {
                    physical_batch_size: 256,
                    elapsed: Duration::from_millis(99),
                },
                default,
            ]),
            512
        );
    }

    #[test]
    fn physical_batch_selection_rejects_output_changes() {
        assert_eq!(
            select_quality_preserving_physical_batch_size(256, false),
            DEFAULT_PHYSICAL_BATCH_SIZE
        );
        assert_eq!(
            select_quality_preserving_physical_batch_size(256, true),
            256
        );
        assert_eq!(
            select_quality_preserving_physical_batch_size(DEFAULT_PHYSICAL_BATCH_SIZE, true),
            DEFAULT_PHYSICAL_BATCH_SIZE
        );
    }

    #[test]
    fn tuning_record_round_trips_and_rejects_a_different_key() {
        let directory =
            std::env::temp_dir().join(format!("trim-prompt-thread-tuning-{}", Uuid::new_v4()));
        let store = ThreadTuningStore::new(directory.clone());
        let configuration = RuntimeInferenceConfig {
            threads: RuntimeThreadCounts {
                generation: 6,
                batch: 7,
            },
            batch_sizes: RuntimeBatchSizes {
                logical: 512,
                physical: 256,
            },
        };

        store
            .store("machine-a", configuration)
            .expect("store tuning");
        assert!(store
            .is_pending_next_launch("machine-a")
            .expect("read pending tuning state"));
        assert_eq!(
            store.load("machine-a", 8).expect("load tuning"),
            Some(configuration)
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
            Some(configuration)
        );

        fs::remove_dir_all(directory).expect("remove tuning test directory");
    }

    #[test]
    fn instruction_profile_requires_all_avx2_engine_features() {
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities {
                sse42: true,
                avx2: true,
                fma: true,
                f16c: true,
                bmi2: true,
                ..Default::default()
            }),
            "avx2-fma-f16c-bmi2"
        );
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities {
                sse42: true,
                avx2: true,
                fma: true,
                f16c: true,
                bmi2: true,
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
                sse42: true,
                avx2: true,
                fma: true,
                f16c: true,
                ..Default::default()
            }),
            "sse4.2"
        );
        assert_eq!(
            cpu_instruction_profile(CpuInstructionCapabilities::default()),
            "unsupported"
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

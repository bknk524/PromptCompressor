use std::error::Error;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::params::LlamaModelParams,
    model::{AddBos, LlamaModel as InnerModel, Special},
    sampling::LlamaSampler,
    LlamaContext,
};

pub(super) use llama_cpp::token::LlamaToken as Token;

type LlamaResult<T> = std::result::Result<T, LlamaError>;

#[derive(Debug)]
pub(super) struct LlamaError(String);

impl Display for LlamaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for LlamaError {}

fn llama_error(context: &str, error: impl Display) -> LlamaError {
    LlamaError(format!("{context}: {error}"))
}

fn backend() -> LlamaResult<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    let initialized = BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init().map_err(|error| error.to_string())?;
        backend.void_logs();
        Ok(backend)
    });
    initialized
        .as_ref()
        .map_err(|error| LlamaError(format!("failed to initialize llama.cpp backend: {error}")))
}

fn expected_cpu_engine() -> &'static str {
    if cfg!(feature = "embedded-llama-avx512") {
        "avx512"
    } else if cfg!(feature = "embedded-llama-avx2") {
        "avx2"
    } else {
        "compatible"
    }
}

fn ensure_compiled_cpu_engine_matches() -> LlamaResult<()> {
    let actual = llama_cpp_sys::TRIMPROMPT_CPU_ENGINE;
    let expected = expected_cpu_engine();
    if actual == expected {
        Ok(())
    } else {
        Err(LlamaError(format!(
            "llama.cpp CPU engine mismatch: application={expected}, library={actual}"
        )))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LlamaParams;

#[derive(Debug, Clone)]
pub(super) struct SessionParams {
    pub(super) n_ctx: u32,
    pub(super) n_threads: u32,
    pub(super) n_threads_batch: u32,
}

impl Default for SessionParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_threads: 1,
            n_threads_batch: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct LlamaModel {
    inner: Arc<InnerModel>,
}

impl LlamaModel {
    pub(super) fn load_from_file(path: &Path, _params: LlamaParams) -> LlamaResult<Self> {
        ensure_compiled_cpu_engine_matches()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = InnerModel::load_from_file(backend()?, path, &params)
            .map_err(|error| llama_error("model load failed", error))?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    pub(super) fn tokenize_bytes(
        &self,
        bytes: &[u8],
        add_bos: bool,
        _special: bool,
    ) -> LlamaResult<Vec<Token>> {
        let text = std::str::from_utf8(bytes)
            .map_err(|error| llama_error("prompt is not valid UTF-8", error))?;
        let add_bos = if add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };
        self.inner
            .str_to_token(text, add_bos)
            .map_err(|error| llama_error("tokenization failed", error))
    }

    pub(super) fn create_session(&self, params: SessionParams) -> LlamaResult<LlamaSession> {
        let context_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(params.n_ctx))
            .with_n_batch(params.n_ctx.clamp(1, 512))
            .with_n_ubatch(params.n_ctx.clamp(1, 512))
            .with_n_threads(params.n_threads.max(1) as i32)
            .with_n_threads_batch(params.n_threads_batch.max(1) as i32);
        let model = Arc::clone(&self.inner);
        let context = model
            .new_context(backend()?, context_params)
            .map_err(|error| llama_error("context creation failed", error))?;

        // contextはArc内のモデルを参照する。SessionInnerがcontextを先に破棄し、modelを
        // 後に保持・破棄するため、このライフタイム延長中も参照先は必ず有効である。
        let context =
            unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context) };
        Ok(LlamaSession {
            inner: Arc::new(Mutex::new(SessionInner {
                context,
                model,
                tokens: Vec::new(),
                last_batch_size: 0,
                params,
            })),
        })
    }
}

#[derive(Debug)]
struct SessionInner {
    // 宣言順に破棄されるため、モデルより先にcontextを解放する。
    context: LlamaContext<'static>,
    model: Arc<InnerModel>,
    tokens: Vec<Token>,
    last_batch_size: usize,
    params: SessionParams,
}

// llama.cpp contextは同時利用せず、必ずMutexを介して単一スレッドから操作する。
unsafe impl Send for SessionInner {}

#[derive(Debug, Clone)]
pub(super) struct LlamaSession {
    inner: Arc<Mutex<SessionInner>>,
}

impl LlamaSession {
    pub(super) fn advance_context(&mut self, bytes: &[u8]) -> LlamaResult<()> {
        let model = {
            let session = self.lock()?;
            LlamaModel {
                inner: Arc::clone(&session.model),
            }
        };
        let tokens = model.tokenize_bytes(bytes, false, true)?;
        self.advance_context_with_tokens(&tokens)
    }

    pub(super) fn advance_context_with_tokens(&mut self, tokens: &[Token]) -> LlamaResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut session = self.lock()?;
        let maximum_batch = session.params.n_ctx.clamp(1, 512) as usize;
        let history_size = session.tokens.len();
        let mut processed = 0usize;

        for chunk in tokens.chunks(maximum_batch) {
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (offset, token) in chunk.iter().copied().enumerate() {
                let absolute = history_size + processed + offset;
                let is_last = absolute + 1 == history_size + tokens.len();
                batch
                    .add(token, absolute as i32, &[0], is_last)
                    .map_err(|error| llama_error("batch creation failed", error))?;
            }
            session
                .context
                .decode(&mut batch)
                .map_err(|error| llama_error("prompt evaluation failed", error))?;
            session.last_batch_size = chunk.len();
            processed += chunk.len();
        }

        session.tokens.extend_from_slice(tokens);
        Ok(())
    }

    pub(super) fn deep_copy(&self) -> LlamaResult<Self> {
        let source = self.lock()?;
        let model = LlamaModel {
            inner: Arc::clone(&source.model),
        };
        let copy = model.create_session(source.params.clone())?;
        let mut destination = copy.lock()?;

        let state_size = source.context.get_state_size();
        let mut state = vec![0_u8; state_size];
        let copied = unsafe { source.context.copy_state_data(state.as_mut_ptr()) };
        if copied > state.len() {
            return Err(LlamaError(format!(
                "llama.cpp state copy exceeded its buffer: {copied} > {}",
                state.len()
            )));
        }
        let restored = unsafe { destination.context.set_state_data(&state[..copied]) };
        if restored != copied {
            return Err(LlamaError(format!(
                "llama.cpp state restore size mismatch: copied={copied}, restored={restored}"
            )));
        }
        destination.tokens.clone_from(&source.tokens);
        destination.last_batch_size = source.last_batch_size;
        drop(destination);
        drop(source);
        Ok(copy)
    }

    pub(super) fn start_completing_with(
        &mut self,
        _sampler: standard_sampler::StandardSampler,
        max_predictions: usize,
    ) -> LlamaResult<CompletionHandle> {
        if self.lock()?.tokens.is_empty() {
            return Err(LlamaError(
                "cannot start completion without prompt history".to_string(),
            ));
        }
        Ok(CompletionHandle {
            session: self.clone(),
            sampler: LlamaSampler::greedy(),
            remaining: max_predictions,
            pending_utf8: Vec::new(),
            finished: false,
        })
    }

    fn lock(&self) -> LlamaResult<std::sync::MutexGuard<'_, SessionInner>> {
        self.inner
            .lock()
            .map_err(|_| LlamaError("llama.cpp session lock is poisoned".to_string()))
    }
}

pub(super) mod standard_sampler {
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct StandardSampler;

    impl StandardSampler {
        pub(crate) fn new_greedy() -> Self {
            Self
        }
    }
}

pub(super) struct CompletionHandle {
    session: LlamaSession,
    sampler: LlamaSampler,
    remaining: usize,
    pending_utf8: Vec<u8>,
    finished: bool,
}

impl CompletionHandle {
    pub(super) fn into_strings(self) -> Self {
        self
    }

    fn finish_pending(&mut self) -> Option<String> {
        if self.pending_utf8.is_empty() {
            None
        } else {
            Some(String::from_utf8_lossy(&std::mem::take(&mut self.pending_utf8)).into_owned())
        }
    }
}

impl Iterator for CompletionHandle {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return self.finish_pending();
        }
        if self.remaining == 0 {
            self.finished = true;
            return self.finish_pending();
        }

        let session_handle = self.session.clone();
        let bytes = {
            let mut session = match session_handle.lock() {
                Ok(session) => session,
                Err(_) => {
                    self.finished = true;
                    return self.finish_pending();
                }
            };
            if session.last_batch_size == 0 {
                self.finished = true;
                return self.finish_pending();
            }

            let logit_index = session.last_batch_size.saturating_sub(1) as i32;
            let token = self.sampler.sample(&session.context, logit_index);
            if token == session.model.token_eos() || token == session.model.token_eot() {
                self.finished = true;
                return self.finish_pending();
            }
            let bytes = match session.model.token_to_raw_bytes(token, Special::Plaintext) {
                Ok(bytes) => bytes,
                Err(_) => {
                    self.finished = true;
                    return self.finish_pending();
                }
            };

            let position = session.tokens.len() as i32;
            let mut batch = LlamaBatch::new(1, 1);
            if batch.add(token, position, &[0], true).is_err()
                || session.context.decode(&mut batch).is_err()
            {
                self.finished = true;
                return self.finish_pending();
            }
            session.tokens.push(token);
            session.last_batch_size = 1;
            bytes
        };

        self.remaining -= 1;
        self.pending_utf8.extend_from_slice(&bytes);
        Some(take_complete_utf8(&mut self.pending_utf8))
    }
}

fn take_complete_utf8(buffer: &mut Vec<u8>) -> String {
    let mut output = String::new();
    loop {
        match std::str::from_utf8(buffer) {
            Ok(text) => {
                output.push_str(text);
                buffer.clear();
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    let prefix = buffer.drain(..valid).collect::<Vec<_>>();
                    output.push_str(std::str::from_utf8(&prefix).expect("validated UTF-8 prefix"));
                }
                if let Some(invalid) = error.error_len() {
                    buffer.drain(..invalid);
                    output.push(char::REPLACEMENT_CHARACTER);
                    continue;
                }
                break;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{ensure_compiled_cpu_engine_matches, take_complete_utf8};

    #[test]
    fn application_and_llama_cpp_cpu_engines_match() {
        ensure_compiled_cpu_engine_matches().expect("CPU engine features must match");
    }

    #[test]
    fn incremental_utf8_waits_for_the_complete_character() {
        let mut bytes = vec![0xe3, 0x81];
        assert_eq!(take_complete_utf8(&mut bytes), "");
        bytes.push(0x82);
        assert_eq!(take_complete_utf8(&mut bytes), "あ");
        assert!(bytes.is_empty());
    }

    #[test]
    fn malformed_utf8_is_replaced_without_losing_following_text() {
        let mut bytes = vec![b'a', 0xff, b'b'];
        assert_eq!(take_complete_utf8(&mut bytes), "a�b");
        assert!(bytes.is_empty());
    }
}

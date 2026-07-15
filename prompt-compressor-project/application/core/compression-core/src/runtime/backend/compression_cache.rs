use std::collections::VecDeque;

use crate::config::profile::ProfileDefinition;
use crate::types::{CompressionRequest, RequestSource};

use super::CompressionDraft;

pub(super) const MAX_COMPRESSION_CACHE_ENTRIES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompressionCacheKey {
    input_text: String,
    compression_level: u8,
    request_profile: String,
    profile_id: String,
    model_ref: String,
    policy_ref: String,
    runtime_ref: String,
    constraints: [bool; 5],
    target_destination: String,
    tokenizer_profile: String,
    source: &'static str,
}

impl CompressionCacheKey {
    pub(super) fn new(request: &CompressionRequest, profile: &ProfileDefinition) -> Self {
        Self {
            input_text: request.input_text.clone(),
            compression_level: request.compression_level.value(),
            request_profile: request.profile.clone(),
            profile_id: profile.id.clone(),
            model_ref: profile.model_ref.clone(),
            policy_ref: profile.policy_ref.clone(),
            runtime_ref: profile.runtime_ref.clone(),
            constraints: [
                request.constraints.preserve_code_blocks,
                request.constraints.preserve_file_names,
                request.constraints.preserve_error_messages,
                request.constraints.preserve_numbers,
                request.constraints.preserve_negations,
            ],
            target_destination: request.target.destination.clone(),
            tokenizer_profile: request.target.tokenizer_profile.clone(),
            source: match request.source {
                RequestSource::Cli => "cli",
                RequestSource::Desktop => "desktop",
            },
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct CompressionDraftCache {
    pub(super) entries: VecDeque<(CompressionCacheKey, CompressionDraft)>,
}

impl CompressionDraftCache {
    pub(super) fn get(&mut self, key: &CompressionCacheKey) -> Option<CompressionDraft> {
        let index = self
            .entries
            .iter()
            .position(|(cached_key, _)| cached_key == key)?;
        let entry = self.entries.remove(index)?;
        let draft = entry.1.clone();
        self.entries.push_back(entry);
        Some(draft)
    }

    pub(super) fn insert(&mut self, key: CompressionCacheKey, draft: CompressionDraft) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(cached_key, _)| cached_key == &key)
        {
            self.entries.remove(index);
        }
        while self.entries.len() >= MAX_COMPRESSION_CACHE_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back((key, draft));
    }
}

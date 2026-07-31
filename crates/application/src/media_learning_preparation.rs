use std::sync::Arc;

use domain::{LanguageCode, MediaId, SubtitleTrackId, TranscriptionJobId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ApplicationError, FoundationPreparationTarget, LearningPreparationRunId};

const MEDIA_LEARNING_PREPARATION_VERSION: &str = "media-learning-preparation-v1";
const INPUTS_CHANGED_ERROR: &str = "media preparation inputs changed";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MediaLearningPreparationId(String);

impl MediaLearningPreparationId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ApplicationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ApplicationError::Validation(
                "media learning preparation identifier",
            ));
        }
        Ok(Self(value))
    }

    pub fn from_fingerprint(fingerprint: &str) -> Self {
        Self(digest_fields(
            "media-learning-preparation-run",
            &[fingerprint],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLearningPreparationTarget {
    pub media_id: MediaId,
    pub media_fingerprint: String,
    pub requested_learning_language: Option<LanguageCode>,
}

impl MediaLearningPreparationTarget {
    pub fn target_key(&self) -> String {
        digest_fields(
            "media-learning-preparation-target",
            &[self.media_id.as_str()],
        )
    }

    fn validate(&self) -> Result<(), ApplicationError> {
        if self.media_fingerprint.trim().is_empty() {
            return Err(ApplicationError::Invalid(
                "media preparation fingerprint must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleTextTrackSnapshot {
    pub media_id: MediaId,
    pub track_id: SubtitleTrackId,
    /// Fingerprint of the imported/raw track identity.
    pub track_fingerprint: String,
    /// Fingerprint of the exact language/text/token snapshot consumed by
    /// foundation preparation.
    pub text_snapshot_fingerprint: String,
    pub language: LanguageCode,
}

impl SubtitleTextTrackSnapshot {
    fn validate(&self) -> Result<(), ApplicationError> {
        if self.track_fingerprint.trim().is_empty()
            || self.text_snapshot_fingerprint.trim().is_empty()
        {
            return Err(ApplicationError::Invalid(
                "subtitle text track fingerprints must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLearningPreparationRequest {
    pub explicit_subtitle_track_id: Option<SubtitleTrackId>,
    pub explicit_audio_track: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
enum ResolvedMediaSubtitleSource {
    Existing { snapshot: SubtitleTextTrackSnapshot },
    Asr { audio_track: Option<u32> },
}

impl ResolvedMediaSubtitleSource {
    fn fingerprint_fields(&self) -> Vec<String> {
        match self {
            Self::Existing { snapshot } => vec![
                "existing".into(),
                snapshot.media_id.as_str().into(),
                snapshot.track_id.as_str().into(),
                snapshot.track_fingerprint.clone(),
                snapshot.text_snapshot_fingerprint.clone(),
                snapshot.language.as_str().into(),
            ],
            Self::Asr { audio_track } => vec![
                "asr".into(),
                audio_track
                    .map(|track| track.to_string())
                    .unwrap_or_else(|| "default".into()),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum MediaLearningPreparationSourceInspection {
    Existing {
        snapshot: SubtitleTextTrackSnapshot,
    },
    Asr {
        audio_track: Option<u32>,
    },
    SelectionRequired {
        reason: MediaLearningPreparationSelectionRequired,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaLearningPreparationSelectionRequired {
    SubtitleTrack,
    AudioTrack,
}

pub trait MediaLearningPreparationInspector: Send + Sync {
    fn inspect(
        &self,
        target: &MediaLearningPreparationTarget,
        request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, ApplicationError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum ReadySubtitleTextTrackSource {
    Existing,
    AsrChild { job_id: TranscriptionJobId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum SubtitleTextTrackSlot {
    Existing {
        snapshot: SubtitleTextTrackSnapshot,
    },
    AsrChild {
        audio_track: Option<u32>,
        job_id: Option<TranscriptionJobId>,
        input_provenance_fingerprint: Option<String>,
    },
    Ready {
        snapshot: SubtitleTextTrackSnapshot,
        source: ReadySubtitleTextTrackSource,
    },
    Failed {
        reason: String,
    },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationPreparationChildRef {
    pub run_id: LearningPreparationRunId,
    pub input_fingerprint: String,
}

impl FoundationPreparationChildRef {
    fn validate(&self) -> Result<(), ApplicationError> {
        if self.input_fingerprint.trim().is_empty() {
            return Err(ApplicationError::Invalid(
                "foundation child input fingerprint must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum FoundationPreparationSlot {
    Pending,
    Child {
        child: FoundationPreparationChildRef,
    },
    Ready {
        child: FoundationPreparationChildRef,
    },
    Failed {
        child: Option<FoundationPreparationChildRef>,
        reason: String,
    },
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaLearningPreparationStatus {
    Queued,
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
}

impl MediaLearningPreparationStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running | Self::Cancelling)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLearningPreparation {
    pub id: MediaLearningPreparationId,
    pub target: MediaLearningPreparationTarget,
    pub request: MediaLearningPreparationRequest,
    source: ResolvedMediaSubtitleSource,
    pub target_key: String,
    pub input_fingerprint: String,
    pub status: MediaLearningPreparationStatus,
    pub subtitle_text_track: SubtitleTextTrackSlot,
    pub foundation: FoundationPreparationSlot,
    pub revision: u64,
    pub retry_of_id: Option<MediaLearningPreparationId>,
    pub error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl MediaLearningPreparation {
    fn new(
        target: MediaLearningPreparationTarget,
        request: MediaLearningPreparationRequest,
        source: ResolvedMediaSubtitleSource,
        now_ms: u64,
    ) -> Result<Self, ApplicationError> {
        target.validate()?;
        validate_resolved_source(&target, &request, &source)?;
        let target_key = target.target_key();
        let input_fingerprint = media_preparation_input_fingerprint(&target, &request, &source);
        let subtitle_text_track = initial_subtitle_slot(&source);
        Ok(Self {
            id: MediaLearningPreparationId::from_fingerprint(&format!(
                "{input_fingerprint}:{now_ms}"
            )),
            target,
            request,
            source,
            target_key,
            input_fingerprint,
            status: MediaLearningPreparationStatus::Queued,
            subtitle_text_track,
            foundation: FoundationPreparationSlot::Pending,
            revision: 0,
            retry_of_id: None,
            error: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        })
    }

    pub fn foundation_target(&self) -> Option<FoundationPreparationTarget> {
        let SubtitleTextTrackSlot::Ready { snapshot, .. } = &self.subtitle_text_track else {
            return None;
        };
        Some(FoundationPreparationTarget {
            media_id: self.target.media_id.clone(),
            media_fingerprint: self.target.media_fingerprint.clone(),
            subtitle_track_id: snapshot.track_id.clone(),
            subtitle_fingerprint: snapshot.track_fingerprint.clone(),
            subtitle_text_fingerprint: snapshot.text_snapshot_fingerprint.clone(),
        })
    }

    pub fn has_valid_identity(&self) -> bool {
        self.target_key == self.target.target_key()
            && self.input_fingerprint
                == media_preparation_input_fingerprint(&self.target, &self.request, &self.source)
    }

    pub fn apply(
        &mut self,
        command: MediaLearningPreparationCommand,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        match command {
            MediaLearningPreparationCommand::Start => self.start(now_ms),
            MediaLearningPreparationCommand::AcceptExistingSubtitle => {
                self.accept_existing_subtitle(now_ms)
            }
            MediaLearningPreparationCommand::AttachAsrChild {
                job_id,
                input_provenance_fingerprint,
            } => self.attach_asr_child(job_id, input_provenance_fingerprint, now_ms),
            MediaLearningPreparationCommand::CompleteAsrChild { job_id, snapshot } => {
                self.complete_asr_child(job_id, snapshot, now_ms)
            }
            MediaLearningPreparationCommand::FailAsrChild { job_id, reason } => {
                self.fail_asr_child(job_id, reason, now_ms)
            }
            MediaLearningPreparationCommand::AttachFoundationChild { child } => {
                self.attach_foundation_child(child, now_ms)
            }
            MediaLearningPreparationCommand::CompleteFoundationChild { run_id } => {
                self.complete_foundation_child(run_id, now_ms)
            }
            MediaLearningPreparationCommand::FailFoundationChild { run_id, reason } => {
                self.fail_foundation_child(run_id, reason, now_ms)
            }
            MediaLearningPreparationCommand::RequestCancel => self.request_cancel(now_ms),
            MediaLearningPreparationCommand::FinishCancellation => self.finish_cancellation(now_ms),
            MediaLearningPreparationCommand::FailExecution { reason } => {
                self.fail_execution(reason, now_ms)
            }
        }
    }

    pub fn recover_after_restart(&mut self, now_ms: u64) {
        if self.status == MediaLearningPreparationStatus::Running {
            self.status = MediaLearningPreparationStatus::Queued;
            self.bump(now_ms);
        }
    }

    fn start(&mut self, now_ms: u64) -> Result<(), ApplicationError> {
        match self.status {
            MediaLearningPreparationStatus::Queued => {
                self.status = MediaLearningPreparationStatus::Running;
                self.bump(now_ms);
                Ok(())
            }
            MediaLearningPreparationStatus::Running => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "media learning preparation is not queued",
            )),
        }
    }

    fn accept_existing_subtitle(&mut self, now_ms: u64) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        match &self.subtitle_text_track {
            SubtitleTextTrackSlot::Existing { snapshot } => {
                self.subtitle_text_track = SubtitleTextTrackSlot::Ready {
                    snapshot: snapshot.clone(),
                    source: ReadySubtitleTextTrackSource::Existing,
                };
                self.bump(now_ms);
                Ok(())
            }
            SubtitleTextTrackSlot::Ready {
                source: ReadySubtitleTextTrackSource::Existing,
                ..
            } => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "media preparation has no existing subtitle to accept",
            )),
        }
    }

    fn attach_asr_child(
        &mut self,
        job_id: TranscriptionJobId,
        input_provenance_fingerprint: String,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        if input_provenance_fingerprint.trim().is_empty() {
            return Err(ApplicationError::Invalid(
                "ASR child input provenance fingerprint must not be empty".into(),
            ));
        }
        let SubtitleTextTrackSlot::AsrChild {
            job_id: current_job_id,
            input_provenance_fingerprint: current_fingerprint,
            ..
        } = &mut self.subtitle_text_track
        else {
            return Err(ApplicationError::Conflict(
                "media preparation does not require an ASR child",
            ));
        };
        match current_job_id {
            None => {
                *current_job_id = Some(job_id);
                *current_fingerprint = Some(input_provenance_fingerprint);
                self.bump(now_ms);
                Ok(())
            }
            Some(current)
                if current == &job_id
                    && current_fingerprint.as_deref()
                        == Some(input_provenance_fingerprint.as_str()) =>
            {
                Ok(())
            }
            Some(_) => Err(ApplicationError::Conflict(
                "media preparation already has another ASR child",
            )),
        }
    }

    fn complete_asr_child(
        &mut self,
        job_id: TranscriptionJobId,
        snapshot: SubtitleTextTrackSnapshot,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        snapshot.validate()?;
        if snapshot.media_id != self.target.media_id
            || self
                .target
                .requested_learning_language
                .as_ref()
                .is_some_and(|requested| requested != &snapshot.language)
        {
            return Err(ApplicationError::Invalid(
                "ASR subtitle snapshot does not match the media preparation target".into(),
            ));
        }
        match &self.subtitle_text_track {
            SubtitleTextTrackSlot::AsrChild {
                job_id: Some(current),
                ..
            } if current == &job_id => {
                self.subtitle_text_track = SubtitleTextTrackSlot::Ready {
                    snapshot,
                    source: ReadySubtitleTextTrackSource::AsrChild { job_id },
                };
                self.bump(now_ms);
                Ok(())
            }
            SubtitleTextTrackSlot::Ready {
                snapshot: current_snapshot,
                source: ReadySubtitleTextTrackSource::AsrChild { job_id: current },
            } if current == &job_id && current_snapshot == &snapshot => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "ASR completion does not match the attached child",
            )),
        }
    }

    fn fail_asr_child(
        &mut self,
        job_id: TranscriptionJobId,
        reason: String,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        validate_reason(&reason)?;
        match &self.subtitle_text_track {
            SubtitleTextTrackSlot::AsrChild {
                job_id: Some(current),
                ..
            } if current == &job_id => {
                self.subtitle_text_track = SubtitleTextTrackSlot::Failed {
                    reason: reason.clone(),
                };
                self.status = MediaLearningPreparationStatus::Failed;
                self.error = Some(reason);
                self.bump(now_ms);
                Ok(())
            }
            SubtitleTextTrackSlot::Failed {
                reason: current_reason,
            } if current_reason == &reason => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "ASR failure does not match the attached child",
            )),
        }
    }

    fn attach_foundation_child(
        &mut self,
        child: FoundationPreparationChildRef,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        child.validate()?;
        if self.foundation_target().is_none() {
            return Err(ApplicationError::Conflict(
                "foundation preparation requires a ready subtitle text track",
            ));
        }
        match &self.foundation {
            FoundationPreparationSlot::Pending => {
                self.foundation = FoundationPreparationSlot::Child { child };
                self.bump(now_ms);
                Ok(())
            }
            FoundationPreparationSlot::Child { child: current } if current == &child => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "media preparation already has another foundation child",
            )),
        }
    }

    fn complete_foundation_child(
        &mut self,
        run_id: LearningPreparationRunId,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        match &self.foundation {
            FoundationPreparationSlot::Child { child } if child.run_id == run_id => {
                self.foundation = FoundationPreparationSlot::Ready {
                    child: child.clone(),
                };
                self.status = MediaLearningPreparationStatus::Completed;
                self.bump(now_ms);
                Ok(())
            }
            FoundationPreparationSlot::Ready { child } if child.run_id == run_id => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "foundation completion does not match the attached child",
            )),
        }
    }

    fn fail_foundation_child(
        &mut self,
        run_id: LearningPreparationRunId,
        reason: String,
        now_ms: u64,
    ) -> Result<(), ApplicationError> {
        ensure_running(self.status)?;
        validate_reason(&reason)?;
        match &self.foundation {
            FoundationPreparationSlot::Child { child } if child.run_id == run_id => {
                self.foundation = FoundationPreparationSlot::Failed {
                    child: Some(child.clone()),
                    reason: reason.clone(),
                };
                self.status = MediaLearningPreparationStatus::Failed;
                self.error = Some(reason);
                self.bump(now_ms);
                Ok(())
            }
            FoundationPreparationSlot::Failed {
                child: Some(child),
                reason: current_reason,
            } if child.run_id == run_id && current_reason == &reason => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "foundation failure does not match the attached child",
            )),
        }
    }

    fn request_cancel(&mut self, now_ms: u64) -> Result<(), ApplicationError> {
        match self.status {
            MediaLearningPreparationStatus::Queued | MediaLearningPreparationStatus::Running => {
                self.status = MediaLearningPreparationStatus::Cancelling;
                self.bump(now_ms);
                Ok(())
            }
            MediaLearningPreparationStatus::Cancelling => Ok(()),
            _ => Err(ApplicationError::Conflict(
                "media learning preparation is already terminal",
            )),
        }
    }

    fn finish_cancellation(&mut self, now_ms: u64) -> Result<(), ApplicationError> {
        if self.status != MediaLearningPreparationStatus::Cancelling {
            return Err(ApplicationError::Conflict(
                "media learning preparation is not cancelling",
            ));
        }
        if !matches!(
            self.subtitle_text_track,
            SubtitleTextTrackSlot::Ready { .. } | SubtitleTextTrackSlot::Failed { .. }
        ) {
            self.subtitle_text_track = SubtitleTextTrackSlot::Cancelled;
        }
        if !matches!(
            self.foundation,
            FoundationPreparationSlot::Ready { .. } | FoundationPreparationSlot::Failed { .. }
        ) {
            self.foundation = FoundationPreparationSlot::Cancelled;
        }
        self.status = MediaLearningPreparationStatus::Cancelled;
        self.bump(now_ms);
        Ok(())
    }

    fn fail_execution(&mut self, reason: String, now_ms: u64) -> Result<(), ApplicationError> {
        if !matches!(
            self.status,
            MediaLearningPreparationStatus::Queued | MediaLearningPreparationStatus::Running
        ) {
            return Err(ApplicationError::Conflict(
                "media learning preparation is not executable",
            ));
        }
        validate_reason(&reason)?;
        if !matches!(
            self.subtitle_text_track,
            SubtitleTextTrackSlot::Ready { .. } | SubtitleTextTrackSlot::Failed { .. }
        ) {
            self.subtitle_text_track = SubtitleTextTrackSlot::Failed {
                reason: reason.clone(),
            };
        }
        if !matches!(
            self.foundation,
            FoundationPreparationSlot::Ready { .. } | FoundationPreparationSlot::Failed { .. }
        ) {
            self.foundation = FoundationPreparationSlot::Failed {
                child: None,
                reason: reason.clone(),
            };
        }
        self.status = MediaLearningPreparationStatus::Failed;
        self.error = Some(reason);
        self.bump(now_ms);
        Ok(())
    }

    fn invalidate(&mut self, now_ms: u64) {
        if !matches!(
            self.subtitle_text_track,
            SubtitleTextTrackSlot::Ready { .. } | SubtitleTextTrackSlot::Failed { .. }
        ) {
            self.subtitle_text_track = SubtitleTextTrackSlot::Cancelled;
        }
        if !matches!(
            self.foundation,
            FoundationPreparationSlot::Ready { .. } | FoundationPreparationSlot::Failed { .. }
        ) {
            self.foundation = FoundationPreparationSlot::Cancelled;
        }
        self.status = MediaLearningPreparationStatus::Failed;
        self.error = Some(INPUTS_CHANGED_ERROR.into());
        self.bump(now_ms);
    }

    fn retry(&self, now_ms: u64) -> Result<Self, ApplicationError> {
        if self.status.is_active()
            || self.status == MediaLearningPreparationStatus::Completed
            || self.error.as_deref() == Some(INPUTS_CHANGED_ERROR)
        {
            return Err(ApplicationError::Conflict(
                "media learning preparation cannot be retried",
            ));
        }
        let subtitle_text_track = match &self.subtitle_text_track {
            SubtitleTextTrackSlot::Ready { .. } => self.subtitle_text_track.clone(),
            _ => initial_subtitle_slot(&self.source),
        };
        Ok(Self {
            id: MediaLearningPreparationId::from_fingerprint(&format!(
                "{}:retry:{now_ms}",
                self.id.as_str()
            )),
            target: self.target.clone(),
            request: self.request.clone(),
            source: self.source.clone(),
            target_key: self.target_key.clone(),
            input_fingerprint: self.input_fingerprint.clone(),
            status: MediaLearningPreparationStatus::Queued,
            subtitle_text_track,
            foundation: FoundationPreparationSlot::Pending,
            revision: 0,
            retry_of_id: Some(self.id.clone()),
            error: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        })
    }

    fn bump(&mut self, now_ms: u64) {
        self.revision += 1;
        self.updated_at_ms = now_ms;
    }
}

fn initial_subtitle_slot(source: &ResolvedMediaSubtitleSource) -> SubtitleTextTrackSlot {
    match source {
        ResolvedMediaSubtitleSource::Existing { snapshot } => SubtitleTextTrackSlot::Existing {
            snapshot: snapshot.clone(),
        },
        ResolvedMediaSubtitleSource::Asr { audio_track } => SubtitleTextTrackSlot::AsrChild {
            audio_track: *audio_track,
            job_id: None,
            input_provenance_fingerprint: None,
        },
    }
}

fn validate_resolved_source(
    target: &MediaLearningPreparationTarget,
    request: &MediaLearningPreparationRequest,
    source: &ResolvedMediaSubtitleSource,
) -> Result<(), ApplicationError> {
    match source {
        ResolvedMediaSubtitleSource::Existing { snapshot } => {
            snapshot.validate()?;
            if snapshot.media_id != target.media_id
                || target
                    .requested_learning_language
                    .as_ref()
                    .is_some_and(|language| language != &snapshot.language)
                || request
                    .explicit_subtitle_track_id
                    .as_ref()
                    .is_some_and(|track_id| track_id != &snapshot.track_id)
            {
                return Err(ApplicationError::Invalid(
                    "resolved subtitle snapshot does not match the media request".into(),
                ));
            }
        }
        ResolvedMediaSubtitleSource::Asr { audio_track } => {
            if request.explicit_subtitle_track_id.is_some()
                || request
                    .explicit_audio_track
                    .is_some_and(|requested| Some(requested) != *audio_track)
            {
                return Err(ApplicationError::Invalid(
                    "resolved ASR source does not match the media request".into(),
                ));
            }
        }
    }
    Ok(())
}

fn media_preparation_input_fingerprint(
    target: &MediaLearningPreparationTarget,
    request: &MediaLearningPreparationRequest,
    source: &ResolvedMediaSubtitleSource,
) -> String {
    let source_fields = source.fingerprint_fields();
    let explicit_subtitle = request
        .explicit_subtitle_track_id
        .as_ref()
        .map(SubtitleTrackId::as_str)
        .unwrap_or("automatic");
    let explicit_audio = request
        .explicit_audio_track
        .map(|track| track.to_string())
        .unwrap_or_else(|| "automatic".into());
    let mut fields = vec![
        target.media_id.as_str(),
        target.media_fingerprint.as_str(),
        target
            .requested_learning_language
            .as_ref()
            .map(LanguageCode::as_str)
            .unwrap_or("unspecified"),
        explicit_subtitle,
        explicit_audio.as_str(),
    ];
    fields.extend(source_fields.iter().map(String::as_str));
    digest_fields(MEDIA_LEARNING_PREPARATION_VERSION, &fields)
}

fn ensure_running(status: MediaLearningPreparationStatus) -> Result<(), ApplicationError> {
    if status != MediaLearningPreparationStatus::Running {
        return Err(ApplicationError::Conflict(
            "media learning preparation is not running",
        ));
    }
    Ok(())
}

fn validate_reason(reason: &str) -> Result<(), ApplicationError> {
    if reason.trim().is_empty() {
        return Err(ApplicationError::Invalid(
            "media preparation failure reason must not be empty".into(),
        ));
    }
    Ok(())
}

fn digest_fields(namespace: &str, fields: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update((namespace.len() as u64).to_be_bytes());
    digest.update(namespace.as_bytes());
    for field in fields {
        digest.update((field.len() as u64).to_be_bytes());
        digest.update(field.as_bytes());
    }
    hex::encode(digest.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaLearningPreparationCommand {
    Start,
    AcceptExistingSubtitle,
    AttachAsrChild {
        job_id: TranscriptionJobId,
        input_provenance_fingerprint: String,
    },
    CompleteAsrChild {
        job_id: TranscriptionJobId,
        snapshot: SubtitleTextTrackSnapshot,
    },
    FailAsrChild {
        job_id: TranscriptionJobId,
        reason: String,
    },
    AttachFoundationChild {
        child: FoundationPreparationChildRef,
    },
    CompleteFoundationChild {
        run_id: LearningPreparationRunId,
    },
    FailFoundationChild {
        run_id: LearningPreparationRunId,
        reason: String,
    },
    RequestCancel,
    FinishCancellation,
    FailExecution {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateMediaLearningPreparation {
    Created(MediaLearningPreparation),
    Existing(MediaLearningPreparation),
    InputChanged(MediaLearningPreparation),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaLearningPreparationTransition {
    Applied(MediaLearningPreparation),
    Rejected(MediaLearningPreparation),
}

pub trait MediaLearningPreparationRepository: Send + Sync {
    fn create_active(
        &self,
        preparation: &MediaLearningPreparation,
    ) -> Result<CreateMediaLearningPreparation, ApplicationError>;
    fn get(
        &self,
        id: &MediaLearningPreparationId,
    ) -> Result<Option<MediaLearningPreparation>, ApplicationError>;
    fn transition(
        &self,
        expected_revision: u64,
        preparation: &MediaLearningPreparation,
    ) -> Result<MediaLearningPreparationTransition, ApplicationError>;
    fn recover_active(
        &self,
        now_ms: u64,
    ) -> Result<Vec<MediaLearningPreparation>, ApplicationError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareMediaLearningResult {
    Run(Box<MediaLearningPreparation>),
    Replaced {
        run: Box<MediaLearningPreparation>,
        /// The last durable state before invalidation, retained so an
        /// orchestrator can cancel any already-attached child side effects.
        invalidated: Box<MediaLearningPreparation>,
    },
    SelectionRequired(MediaLearningPreparationSelectionRequired),
    Unavailable(String),
}

pub struct MediaLearningPreparationUseCases {
    preparations: Arc<dyn MediaLearningPreparationRepository>,
    inspector: Arc<dyn MediaLearningPreparationInspector>,
}

impl MediaLearningPreparationUseCases {
    pub fn new(
        preparations: Arc<dyn MediaLearningPreparationRepository>,
        inspector: Arc<dyn MediaLearningPreparationInspector>,
    ) -> Self {
        Self {
            preparations,
            inspector,
        }
    }

    pub fn prepare(
        &self,
        target: MediaLearningPreparationTarget,
        request: MediaLearningPreparationRequest,
        now_ms: u64,
    ) -> Result<PrepareMediaLearningResult, ApplicationError> {
        target.validate()?;
        let source = self.inspector.inspect(&target, &request)?;
        self.prepare_resolved(target, request, source, now_ms)
    }

    pub fn prepare_resolved(
        &self,
        target: MediaLearningPreparationTarget,
        request: MediaLearningPreparationRequest,
        source: MediaLearningPreparationSourceInspection,
        now_ms: u64,
    ) -> Result<PrepareMediaLearningResult, ApplicationError> {
        target.validate()?;
        let source = match source {
            MediaLearningPreparationSourceInspection::Existing { snapshot } => {
                ResolvedMediaSubtitleSource::Existing { snapshot }
            }
            MediaLearningPreparationSourceInspection::Asr { audio_track } => {
                ResolvedMediaSubtitleSource::Asr { audio_track }
            }
            MediaLearningPreparationSourceInspection::SelectionRequired { reason } => {
                return Ok(PrepareMediaLearningResult::SelectionRequired(reason));
            }
            MediaLearningPreparationSourceInspection::Unavailable { reason } => {
                return Ok(PrepareMediaLearningResult::Unavailable(reason));
            }
        };
        let preparation = MediaLearningPreparation::new(target, request, source, now_ms)?;
        match self.preparations.create_active(&preparation)? {
            CreateMediaLearningPreparation::Created(run)
            | CreateMediaLearningPreparation::Existing(run) => {
                Ok(PrepareMediaLearningResult::Run(Box::new(run)))
            }
            CreateMediaLearningPreparation::InputChanged(mut previous) => {
                let invalidated = previous.clone();
                let expected_revision = previous.revision;
                previous.invalidate(now_ms);
                match self.preparations.transition(expected_revision, &previous)? {
                    MediaLearningPreparationTransition::Applied(_) => {}
                    MediaLearningPreparationTransition::Rejected(_) => {
                        return Err(ApplicationError::Conflict(
                            "media learning preparation changed concurrently",
                        ));
                    }
                }
                match self.preparations.create_active(&preparation)? {
                    CreateMediaLearningPreparation::Created(run)
                    | CreateMediaLearningPreparation::Existing(run) => {
                        Ok(PrepareMediaLearningResult::Replaced {
                            run: Box::new(run),
                            invalidated: Box::new(invalidated),
                        })
                    }
                    CreateMediaLearningPreparation::InputChanged(_) => Err(
                        ApplicationError::Conflict("media preparation inputs changed concurrently"),
                    ),
                }
            }
        }
    }

    pub fn get(
        &self,
        id: &MediaLearningPreparationId,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        self.preparations
            .get(id)?
            .ok_or(ApplicationError::NotFound("media learning preparation"))
    }

    pub fn command(
        &self,
        id: &MediaLearningPreparationId,
        command: MediaLearningPreparationCommand,
        now_ms: u64,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        let mut current = self.get(id)?;
        loop {
            let expected_revision = current.revision;
            current.apply(command.clone(), now_ms)?;
            if current.revision == expected_revision {
                return Ok(current);
            }
            match self.preparations.transition(expected_revision, &current)? {
                MediaLearningPreparationTransition::Applied(run) => return Ok(run),
                MediaLearningPreparationTransition::Rejected(run) => current = run,
            }
        }
    }

    pub fn retry(
        &self,
        id: &MediaLearningPreparationId,
        now_ms: u64,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        let previous = self.get(id)?;
        let retry = previous.retry(now_ms)?;
        match self.preparations.create_active(&retry)? {
            CreateMediaLearningPreparation::Created(run)
            | CreateMediaLearningPreparation::Existing(run) => Ok(run),
            CreateMediaLearningPreparation::InputChanged(_) => Err(ApplicationError::Conflict(
                "media preparation inputs changed before retry",
            )),
        }
    }

    pub fn recover_startup(
        &self,
        now_ms: u64,
    ) -> Result<Vec<MediaLearningPreparation>, ApplicationError> {
        self.preparations.recover_active(now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> MediaLearningPreparationTarget {
        MediaLearningPreparationTarget {
            media_id: MediaId::parse("media").unwrap(),
            media_fingerprint: "media-fp".into(),
            requested_learning_language: Some(LanguageCode::parse("en").unwrap()),
        }
    }

    fn snapshot() -> SubtitleTextTrackSnapshot {
        SubtitleTextTrackSnapshot {
            media_id: MediaId::parse("media").unwrap(),
            track_id: SubtitleTrackId::parse("track").unwrap(),
            track_fingerprint: "raw-track".into(),
            text_snapshot_fingerprint: "exact-text-snapshot".into(),
            language: LanguageCode::parse("en").unwrap(),
        }
    }

    fn request(
        explicit_subtitle_track_id: Option<SubtitleTrackId>,
        explicit_audio_track: Option<u32>,
    ) -> MediaLearningPreparationRequest {
        MediaLearningPreparationRequest {
            explicit_subtitle_track_id,
            explicit_audio_track,
        }
    }

    #[test]
    fn foundation_child_cannot_start_until_exact_subtitle_snapshot_is_ready() {
        let mut preparation = MediaLearningPreparation::new(
            target(),
            request(None, None),
            ResolvedMediaSubtitleSource::Existing {
                snapshot: snapshot(),
            },
            1,
        )
        .unwrap();
        preparation
            .apply(MediaLearningPreparationCommand::Start, 2)
            .unwrap();
        let child = FoundationPreparationChildRef {
            run_id: LearningPreparationRunId::parse("foundation").unwrap(),
            input_fingerprint: "foundation-input".into(),
        };

        assert!(
            preparation
                .apply(
                    MediaLearningPreparationCommand::AttachFoundationChild {
                        child: child.clone()
                    },
                    3
                )
                .is_err()
        );
        preparation
            .apply(MediaLearningPreparationCommand::AcceptExistingSubtitle, 4)
            .unwrap();
        preparation
            .apply(
                MediaLearningPreparationCommand::AttachFoundationChild { child },
                5,
            )
            .unwrap();
        assert_eq!(
            preparation
                .foundation_target()
                .unwrap()
                .subtitle_text_fingerprint,
            "exact-text-snapshot"
        );
    }

    #[test]
    fn existing_snapshot_must_match_the_requested_learning_language() {
        let mut wrong_language = snapshot();
        wrong_language.language = LanguageCode::parse("zh").unwrap();

        let result = MediaLearningPreparation::new(
            target(),
            request(None, None),
            ResolvedMediaSubtitleSource::Existing {
                snapshot: wrong_language,
            },
            1,
        );

        assert!(matches!(result, Err(ApplicationError::Invalid(_))));
    }

    #[test]
    fn asr_child_must_be_attached_before_it_can_freeze_a_subtitle_snapshot() {
        let mut preparation = MediaLearningPreparation::new(
            target(),
            request(None, Some(2)),
            ResolvedMediaSubtitleSource::Asr {
                audio_track: Some(2),
            },
            1,
        )
        .unwrap();
        preparation
            .apply(MediaLearningPreparationCommand::Start, 2)
            .unwrap();
        let job_id = TranscriptionJobId::parse("asr").unwrap();

        assert!(
            preparation
                .apply(
                    MediaLearningPreparationCommand::CompleteAsrChild {
                        job_id: job_id.clone(),
                        snapshot: snapshot(),
                    },
                    3,
                )
                .is_err()
        );
        preparation
            .apply(
                MediaLearningPreparationCommand::AttachAsrChild {
                    job_id: job_id.clone(),
                    input_provenance_fingerprint: "asr-input-provenance".into(),
                },
                4,
            )
            .unwrap();
        preparation
            .apply(
                MediaLearningPreparationCommand::CompleteAsrChild {
                    job_id,
                    snapshot: snapshot(),
                },
                5,
            )
            .unwrap();
        assert!(preparation.foundation_target().is_some());
    }

    #[test]
    fn asr_snapshot_must_match_parent_media_and_requested_language() {
        let mut preparation = MediaLearningPreparation::new(
            target(),
            request(None, Some(2)),
            ResolvedMediaSubtitleSource::Asr {
                audio_track: Some(2),
            },
            1,
        )
        .unwrap();
        preparation
            .apply(MediaLearningPreparationCommand::Start, 2)
            .unwrap();
        let job_id = TranscriptionJobId::parse("asr").unwrap();
        preparation
            .apply(
                MediaLearningPreparationCommand::AttachAsrChild {
                    job_id: job_id.clone(),
                    input_provenance_fingerprint: "asr-input-provenance".into(),
                },
                3,
            )
            .unwrap();

        let mut wrong_media = snapshot();
        wrong_media.media_id = MediaId::parse("other-media").unwrap();
        assert!(
            preparation
                .apply(
                    MediaLearningPreparationCommand::CompleteAsrChild {
                        job_id: job_id.clone(),
                        snapshot: wrong_media,
                    },
                    4,
                )
                .is_err()
        );

        let mut wrong_language = snapshot();
        wrong_language.language = LanguageCode::parse("zh").unwrap();
        assert!(
            preparation
                .apply(
                    MediaLearningPreparationCommand::CompleteAsrChild {
                        job_id,
                        snapshot: wrong_language,
                    },
                    5,
                )
                .is_err()
        );
        assert!(matches!(
            preparation.subtitle_text_track,
            SubtitleTextTrackSlot::AsrChild { .. }
        ));
    }

    #[test]
    fn restart_requeues_parent_without_discarding_durable_child_reference() {
        let mut preparation = MediaLearningPreparation::new(
            target(),
            request(None, None),
            ResolvedMediaSubtitleSource::Asr { audio_track: None },
            1,
        )
        .unwrap();
        let job_id = TranscriptionJobId::parse("asr").unwrap();
        preparation
            .apply(MediaLearningPreparationCommand::Start, 2)
            .unwrap();
        preparation
            .apply(
                MediaLearningPreparationCommand::AttachAsrChild {
                    job_id: job_id.clone(),
                    input_provenance_fingerprint: "asr-input-provenance".into(),
                },
                3,
            )
            .unwrap();

        preparation.recover_after_restart(4);

        assert_eq!(preparation.status, MediaLearningPreparationStatus::Queued);
        assert!(matches!(
            preparation.subtitle_text_track,
            SubtitleTextTrackSlot::AsrChild {
                job_id: Some(ref current),
                ..
            } if current == &job_id
        ));
    }
}

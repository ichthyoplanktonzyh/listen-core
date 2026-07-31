use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use api_events::{EventEnvelope, EventName};
use application::{
    AppServices, ApplicationError, CreateTranscriptionJob, ImportSubtitle,
    TranscriptionJobTransition, TranscriptionRepository, now_ms,
};
use domain::{
    MediaAvailability, MediaId, RecordingTranscriptProvenance, RecordingTranscriptionJob,
    RecordingTranscriptionJobId, RecordingTranscriptionStatus, SubtitleTrackProvenance,
    SubtitleTrackStatus, TranscriptionDestination, TranscriptionJob, TranscriptionJobId,
    TranscriptionJobStatus, TranscriptionModelDescriptor, TranscriptionModelId,
    TranscriptionModelState, TranscriptionProviderInfo, TranscriptionPurpose, TranscriptionQuality,
    TranscriptionSegment,
};
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, broadcast};

use crate::download::{ArtifactDownloader, DownloadProgress, ReqwestArtifactDownloader};
use crate::process::{
    CancellationProbe, NeverCancelled, ProcessOutputObserver, ProcessRunner, ProcessSpec,
    TokioProcessRunner,
};
use crate::runtime_support::{
    ffmpeg_wav_args, file_id, hash_file, io_error, resolve_tool, support_dir,
};

#[derive(Clone)]
pub struct TranscriptionCoordinator {
    services: AppServices,
    repository: Arc<dyn TranscriptionRepository>,
    events: broadcast::Sender<EventEnvelope>,
    queue: Arc<Semaphore>,
    model_dir: PathBuf,
    temp_dir: PathBuf,
    process_runner: Arc<dyn ProcessRunner>,
    downloader: Arc<dyn ArtifactDownloader>,
    ffprobe: Option<PathBuf>,
    recommended_model_install: Arc<tokio::sync::Mutex<()>>,
    recording_jobs: Arc<Mutex<HashMap<RecordingTranscriptionJobId, RecordingTranscriptionJob>>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CreateJobRequest {
    pub media_id: String,
    pub model_id: String,
    pub destination: TranscriptionDestination,
    pub purpose: TranscriptionPurpose,
    pub language: Option<String>,
    pub audio_track: Option<u32>,
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CreateRecordingTranscriptionRequest {
    pub recording_id: String,
    pub model_id: String,
    pub language: Option<String>,
}

/// Exact, provider-neutral input for an ASR child owned by the internal
/// media-preparation journey. The parent resolves audio ambiguity before
/// calling this seam; the ASR operation always transcribes the selected source
/// language and never claims subtitle-selection authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EnsurePreparationTranscriptionRequest {
    pub(crate) idempotency_key: String,
    pub(crate) child_id: TranscriptionJobId,
    pub(crate) media_id: MediaId,
    pub(crate) language: Option<String>,
    pub(crate) audio_track: u32,
    pub(crate) terminal_policy: PreparationTranscriptionTerminalPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparationTranscriptionTerminalPolicy {
    Preserve,
    Restart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparationTranscriptionDisposition {
    Created,
    ExistingActive,
    ReusedCompleted,
    RestartedInterrupted,
    RestartedInvalidCompletion,
    RestartedExplicitRetry,
    ExistingTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EnsuredPreparationTranscription {
    pub(crate) disposition: PreparationTranscriptionDisposition,
    pub(crate) job: TranscriptionJob,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PreparationAudioSelection {
    Selected { audio_track: u32 },
    SelectionRequired { reason: &'static str },
    Unavailable { reason: &'static str },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PreparationTranscriptionModelSelection {
    Ready { model_id: TranscriptionModelId },
    InstallRecommended { model_id: TranscriptionModelId },
}

pub(crate) fn preparation_transcription_child_id(
    idempotency_key: &str,
) -> Result<TranscriptionJobId, ApplicationError> {
    if idempotency_key.trim().is_empty() {
        return Err(ApplicationError::Validation(
            "preparation transcription idempotency key",
        ));
    }
    Ok(TranscriptionJobId::from_fingerprint(
        "preparation-transcription-child",
        idempotency_key,
    ))
}

pub(crate) fn resolve_preparation_audio_selection(
    audio_track_count: usize,
    requested_audio_track: Option<u32>,
) -> PreparationAudioSelection {
    match (audio_track_count, requested_audio_track) {
        (0, _) => PreparationAudioSelection::Unavailable {
            reason: "audio_track_not_found",
        },
        (1, None) => PreparationAudioSelection::Selected { audio_track: 0 },
        (count, None) if count > 1 => PreparationAudioSelection::SelectionRequired {
            reason: "audio_track_required",
        },
        (count, Some(audio_track)) if (audio_track as usize) < count => {
            PreparationAudioSelection::Selected { audio_track }
        }
        (_, Some(_)) => PreparationAudioSelection::Unavailable {
            reason: "audio_track_not_found",
        },
        _ => unreachable!("all audio-track selection states are covered"),
    }
}

impl TranscriptionCoordinator {
    pub fn new(
        services: AppServices,
        repository: Arc<dyn TranscriptionRepository>,
        events: broadcast::Sender<EventEnvelope>,
    ) -> Result<Self, ApplicationError> {
        Self::new_with_adapters(
            services,
            repository,
            events,
            Arc::new(TokioProcessRunner),
            Arc::new(ReqwestArtifactDownloader),
        )
    }

    pub fn new_with_process_runner(
        services: AppServices,
        repository: Arc<dyn TranscriptionRepository>,
        events: broadcast::Sender<EventEnvelope>,
        process_runner: Arc<dyn ProcessRunner>,
    ) -> Result<Self, ApplicationError> {
        Self::new_with_adapters(
            services,
            repository,
            events,
            process_runner,
            Arc::new(ReqwestArtifactDownloader),
        )
    }

    pub fn new_with_adapters(
        services: AppServices,
        repository: Arc<dyn TranscriptionRepository>,
        events: broadcast::Sender<EventEnvelope>,
        process_runner: Arc<dyn ProcessRunner>,
        downloader: Arc<dyn ArtifactDownloader>,
    ) -> Result<Self, ApplicationError> {
        Self::new_with_adapters_and_ffprobe(
            services,
            repository,
            events,
            process_runner,
            downloader,
            resolve_tool("LLPLAYERNEXT_FFPROBE", "ffprobe"),
        )
    }

    fn new_with_adapters_and_ffprobe(
        services: AppServices,
        repository: Arc<dyn TranscriptionRepository>,
        events: broadcast::Sender<EventEnvelope>,
        process_runner: Arc<dyn ProcessRunner>,
        downloader: Arc<dyn ArtifactDownloader>,
        ffprobe: Option<PathBuf>,
    ) -> Result<Self, ApplicationError> {
        let support = support_dir();
        let value = Self {
            services,
            repository,
            events,
            queue: Arc::new(Semaphore::new(1)),
            model_dir: support.join("models/transcription/whisper.cpp"),
            temp_dir: std::env::temp_dir().join("LLPlayerNext/transcription"),
            process_runner,
            downloader,
            ffprobe,
            recommended_model_install: Arc::new(tokio::sync::Mutex::new(())),
            recording_jobs: Arc::default(),
        };
        value.repository.interrupt_active_jobs(now_ms())?;
        // Best-effort cleanup of stale work directories left behind when a
        // previous run exited before the detached sound-line task could remove
        // them. Safe at startup because no jobs are running yet.
        let _ = std::fs::remove_dir_all(&value.temp_dir);
        value.seed_catalog()?;
        Ok(value)
    }

    pub fn providers(&self) -> Vec<TranscriptionProviderInfo> {
        let runtime = resolve_tool("LLPLAYERNEXT_WHISPER_CLI", "whisper-cli");
        vec![TranscriptionProviderInfo {
            id: "whisper.cpp".into(),
            display_name: "whisper.cpp".into(),
            runtime_id: "whisper.cpp-cli".into(),
            runtime_version: runtime
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or("unavailable")
                .into(),
            available: runtime.is_some(),
            supports_translation: true,
            supported_languages: vec!["auto".into(), "en".into(), "multilingual".into()],
            diagnostic: runtime
                .is_none()
                .then_some("Bundled whisper-cli was not found.".into()),
        }]
    }

    pub fn models(&self) -> Result<Vec<TranscriptionModelDescriptor>, ApplicationError> {
        self.repository.list_models()
    }

    pub fn jobs(&self) -> Result<Vec<TranscriptionJob>, ApplicationError> {
        self.repository.list_jobs()
    }

    pub fn job(
        &self,
        id: &TranscriptionJobId,
    ) -> Result<Option<TranscriptionJob>, ApplicationError> {
        self.repository.get_job(id)
    }

    pub fn recording_transcription_job(
        &self,
        id: &RecordingTranscriptionJobId,
    ) -> Option<RecordingTranscriptionJob> {
        self.recording_jobs
            .lock()
            .expect("recording transcription jobs mutex poisoned")
            .get(id)
            .cloned()
    }

    pub fn create_recording_transcription(
        self: Arc<Self>,
        request: CreateRecordingTranscriptionRequest,
    ) -> Result<RecordingTranscriptionJob, ApplicationError> {
        let recording_id = domain::RecordingAssetId::parse(request.recording_id)?;
        let recording = self
            .services
            .recordings()
            .recording_asset(&recording_id)?
            .ok_or(ApplicationError::NotFound("recording asset"))?;
        validate_recording_input(&recording)?;
        let model_id = TranscriptionModelId::parse(request.model_id)?;
        let model = self
            .repository
            .get_model(&model_id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        if !matches!(
            model.state,
            TranscriptionModelState::Installed | TranscriptionModelState::Custom
        ) {
            return Err(ApplicationError::Validation(
                "installed transcription model",
            ));
        }
        let requested_language = request
            .language
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .or_else(|| Some(recording.language.as_str().to_owned()));
        if model.english_only
            && requested_language
                .as_deref()
                .is_some_and(|value| value != "en" && value != "auto")
        {
            return Err(ApplicationError::Validation(
                "English recording language for English-only model",
            ));
        }
        let provider = self.providers().into_iter().next().expect("first provider");
        if !provider.available {
            return Err(ApplicationError::Validation("transcription runtime"));
        }
        let created_at_ms = now_ms();
        let id = RecordingTranscriptionJobId::from_fingerprint(
            "recording-transcription-job",
            &format!(
                "{}:{}:{}:{created_at_ms}",
                recording.id.as_str(),
                model.id.as_str(),
                requested_language.as_deref().unwrap_or("auto")
            ),
        );
        let job = RecordingTranscriptionJob {
            id: id.clone(),
            recording_asset_id: recording.id,
            status: RecordingTranscriptionStatus::Queued,
            raw_transcript: None,
            segments: Vec::new(),
            provenance: RecordingTranscriptProvenance {
                provider_id: provider.id,
                provider_version: provider.runtime_version.clone(),
                runtime_id: provider.runtime_id,
                runtime_version: provider.runtime_version,
                model_id: model.id,
                model_revision: model.revision,
                model_checksum_sha256: model.checksum_sha256,
                recording_content_sha256: recording.audio.content_sha256,
                requested_language,
                detected_language: None,
            },
            error_code: None,
            error_message: None,
            created_at_ms,
            started_at_ms: None,
            completed_at_ms: None,
            latency_ms: None,
        };
        self.recording_jobs
            .lock()
            .expect("recording transcription jobs mutex poisoned")
            .insert(id.clone(), job.clone());
        tokio::spawn(async move { self.run_recording_transcription(id).await });
        Ok(job)
    }

    pub fn cancel_recording_transcription(
        &self,
        id: &RecordingTranscriptionJobId,
    ) -> Result<RecordingTranscriptionJob, ApplicationError> {
        let mut jobs = self
            .recording_jobs
            .lock()
            .expect("recording transcription jobs mutex poisoned");
        let job = jobs
            .get_mut(id)
            .ok_or(ApplicationError::NotFound("recording transcription job"))?;
        if !matches!(
            job.status,
            RecordingTranscriptionStatus::Completed
                | RecordingTranscriptionStatus::Failed
                | RecordingTranscriptionStatus::Cancelled
        ) {
            let completed_at_ms = now_ms();
            job.status = RecordingTranscriptionStatus::Cancelled;
            job.completed_at_ms = Some(completed_at_ms);
            job.latency_ms = Some(completed_at_ms.saturating_sub(job.created_at_ms));
        }
        Ok(job.clone())
    }

    pub async fn install_model(
        self: Arc<Self>,
        id: TranscriptionModelId,
    ) -> Result<TranscriptionModelDescriptor, ApplicationError> {
        let mut model = self
            .repository
            .get_model(&id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        let url = model
            .download_url
            .clone()
            .ok_or(ApplicationError::Validation("model download URL"))?;
        tokio::fs::create_dir_all(&self.model_dir)
            .await
            .map_err(io_error)?;
        let path = self.model_dir.join(format!("{}.bin", file_id(id.as_str())));
        let partial = path.with_extension("bin.partial");
        model.state = TranscriptionModelState::Installing;
        model.error = None;
        model.updated_at_ms = now_ms();
        self.repository.upsert_model(&model)?;
        self.emit(EventName::TranscriptionModelChanged, &model);

        let result = async {
            self.downloader
                .download(
                    &url,
                    &partial,
                    Arc::new(TranscriptionDownloadProgress {
                        repository: self.repository.clone(),
                        events: self.events.clone(),
                        model_id: id.clone(),
                    }),
                )
                .await?;
            let checksum = hash_file(&partial)?;
            if checksum != model.checksum_sha256 {
                return Err(ApplicationError::Repository(
                    "model checksum mismatch".into(),
                ));
            }
            tokio::fs::rename(&partial, &path).await.map_err(io_error)?;
            Ok::<_, ApplicationError>(())
        }
        .await;

        match result {
            Ok(()) => {
                model.state = TranscriptionModelState::Installed;
                model.local_path = Some(path.to_string_lossy().into_owned());
                model.installed_bytes = model.size_bytes;
            }
            Err(error) => {
                let _ = tokio::fs::remove_file(&partial).await;
                if let Some(current) = self.repository.get_model(&id)?
                    && current.state == TranscriptionModelState::Downloadable
                {
                    return Ok(current);
                }
                model.state = TranscriptionModelState::Failed;
                model.error = Some(error.to_string());
            }
        }
        model.updated_at_ms = now_ms();
        let model = self.repository.upsert_model(&model)?;
        self.emit(EventName::TranscriptionModelChanged, &model);
        Ok(model)
    }

    pub fn register_custom_model(
        &self,
        path: String,
    ) -> Result<TranscriptionModelDescriptor, ApplicationError> {
        let metadata = std::fs::metadata(&path)
            .map_err(|error| ApplicationError::Repository(error.to_string()))?;
        let checksum = hash_file(Path::new(&path))?;
        let id = TranscriptionModelId::from_fingerprint("custom-transcription-model", &checksum);
        let english_only = Path::new(&path)
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(english_only_model_name);
        self.repository.upsert_model(&TranscriptionModelDescriptor {
            id,
            provider_id: "whisper.cpp".into(),
            display_name: Path::new(&path)
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("Custom model")
                .into(),
            family: "custom".into(),
            revision: checksum.clone(),
            checksum_sha256: checksum,
            download_url: None,
            local_path: Some(path),
            size_bytes: metadata.len(),
            quality: TranscriptionQuality::Balanced,
            english_only,
            supports_translation: !english_only,
            state: TranscriptionModelState::Custom,
            installed_bytes: metadata.len(),
            error: None,
            license: "User supplied".into(),
            updated_at_ms: now_ms(),
        })
    }

    pub async fn delete_model(&self, id: &TranscriptionModelId) -> Result<(), ApplicationError> {
        if self.repository.list_jobs()?.iter().any(|job| {
            job.model_id == *id
                && matches!(
                    job.status,
                    TranscriptionJobStatus::Queued
                        | TranscriptionJobStatus::Extracting
                        | TranscriptionJobStatus::Transcribing
                        | TranscriptionJobStatus::Importing
                )
        }) {
            return Err(ApplicationError::Validation(
                "model is used by an active job",
            ));
        }
        if let Some(mut model) = self.repository.get_model(id)?
            && model.state == TranscriptionModelState::Installed
            && let Some(path) = model.local_path.clone()
        {
            let _ = tokio::fs::remove_file(path).await;
            model.local_path = None;
            model.installed_bytes = 0;
            model.state = TranscriptionModelState::Downloadable;
            model.error = None;
            model.updated_at_ms = now_ms();
            self.repository.upsert_model(&model)?;
            self.emit(EventName::TranscriptionModelChanged, &model);
            return Ok(());
        }
        self.repository.delete_model(id)
    }

    pub fn cancel_model_install(
        &self,
        id: &TranscriptionModelId,
    ) -> Result<TranscriptionModelDescriptor, ApplicationError> {
        let mut model = self
            .repository
            .get_model(id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        if model.state == TranscriptionModelState::Installing {
            model.state = TranscriptionModelState::Downloadable;
            model.installed_bytes = 0;
            model.error = Some("Installation cancelled by user.".into());
            model.updated_at_ms = now_ms();
            self.repository.upsert_model(&model)?;
            self.emit(EventName::TranscriptionModelChanged, &model);
        }
        Ok(model)
    }

    pub fn create_job(
        self: Arc<Self>,
        request: CreateJobRequest,
    ) -> Result<TranscriptionJob, ApplicationError> {
        self.create_job_with_retry(request, None)
    }

    fn create_job_with_retry(
        self: Arc<Self>,
        request: CreateJobRequest,
        retry_of_job_id: Option<TranscriptionJobId>,
    ) -> Result<TranscriptionJob, ApplicationError> {
        let force = request.force;
        let job = self.build_job_candidate(request, retry_of_job_id, None)?;
        if !force && let Some(job) = self.repository.find_completed_job(&job.input_fingerprint)? {
            return Ok(job);
        }
        let job = self.repository.create_job(&job)?;
        self.emit(EventName::TranscriptionJobChanged, &job);
        self.clone().start_job(job.id.clone());
        Ok(job)
    }

    /// Probe the exact media before ASR so a missing or ambiguous audio stream
    /// becomes a typed preparation result instead of an ffmpeg default.
    pub(crate) async fn resolve_preparation_audio_track(
        &self,
        media_id: &MediaId,
        requested_audio_track: Option<u32>,
    ) -> Result<PreparationAudioSelection, ApplicationError> {
        let media = self
            .services
            .media_analysis()
            .read_media(media_id)?
            .ok_or(ApplicationError::NotFound("media"))?;
        if media.availability != MediaAvailability::Available {
            return Ok(PreparationAudioSelection::Unavailable {
                reason: "media_unavailable",
            });
        }
        let Some(ffprobe) = self.ffprobe.clone() else {
            return Ok(PreparationAudioSelection::Unavailable {
                reason: "audio_track_probe_failed",
            });
        };
        let counter = Arc::new(PreparationAudioTrackCounter::default());
        if self
            .process_runner
            .run_streaming(
                ProcessSpec::new(
                    ffprobe,
                    vec![
                        "-v".into(),
                        "error".into(),
                        "-select_streams".into(),
                        "a".into(),
                        "-show_entries".into(),
                        "stream=index".into(),
                        "-of".into(),
                        "csv=p=0".into(),
                        media.path,
                    ],
                ),
                Arc::new(NeverCancelled),
                counter.clone(),
            )
            .await
            .is_err()
        {
            return Ok(PreparationAudioSelection::Unavailable {
                reason: "audio_track_probe_failed",
            });
        }
        Ok(resolve_preparation_audio_selection(
            counter.count(),
            requested_audio_track,
        ))
    }

    /// Create or recover the deterministic ASR child owned by an internal
    /// media-preparation run. Public transcription endpoints keep their
    /// existing timestamped create/retry behavior; only this seam reuses a
    /// caller-chosen child identity and restarts interrupted work in place.
    pub(crate) async fn ensure_preparation_transcription(
        self: &Arc<Self>,
        request: EnsurePreparationTranscriptionRequest,
    ) -> Result<EnsuredPreparationTranscription, ApplicationError> {
        let expected_id = preparation_transcription_child_id(&request.idempotency_key)?;
        if request.child_id != expected_id {
            return Err(ApplicationError::Invalid(
                "preparation transcription child ID does not match its idempotency key".into(),
            ));
        }
        if let Some(existing) = self.repository.get_job(&request.child_id)? {
            self.validate_existing_preparation_job(&existing, &request)?;
            let (ensured, should_start) =
                self.claim_existing_preparation_job(existing, request.terminal_policy)?;
            if should_start {
                self.clone().start_job(ensured.job.id.clone());
            }
            return Ok(ensured);
        }
        if !self
            .providers()
            .into_iter()
            .any(|provider| provider.available)
        {
            return Err(ApplicationError::Validation("transcription runtime"));
        }
        let model_id = self
            .ensure_recommended_preparation_model(request.language.as_deref())
            .await?;
        let candidate = self.build_job_candidate(
            CreateJobRequest {
                media_id: request.media_id.as_str().to_owned(),
                model_id: model_id.as_str().to_owned(),
                destination: TranscriptionDestination::Primary,
                purpose: TranscriptionPurpose::Transcribe,
                language: request.language,
                audio_track: Some(request.audio_track),
                force: false,
            },
            None,
            Some(request.child_id),
        )?;
        let (ensured, should_start) =
            self.claim_preparation_job(candidate, request.terminal_policy)?;
        if should_start {
            self.clone().start_job(ensured.job.id.clone());
        }
        Ok(ensured)
    }

    fn validate_existing_preparation_job(
        &self,
        job: &TranscriptionJob,
        request: &EnsurePreparationTranscriptionRequest,
    ) -> Result<(), ApplicationError> {
        let media = self
            .services
            .media_analysis()
            .read_media(&request.media_id)?
            .ok_or(ApplicationError::NotFound("media"))?;
        if job.id != request.child_id
            || job.media_id != request.media_id
            || job.media_fingerprint != media.fingerprint
            || job.destination != TranscriptionDestination::Primary
            || job.purpose != TranscriptionPurpose::Transcribe
            || job.requested_language != request.language
            || job.audio_track != Some(request.audio_track)
            || job.input_fingerprint.trim().is_empty()
            || job.archived_at_ms.is_some()
        {
            return Err(ApplicationError::Conflict(
                "preparation transcription idempotency key is already bound to different inputs",
            ));
        }
        Ok(())
    }

    pub(crate) fn recommended_preparation_model(
        &self,
        language: Option<&str>,
    ) -> Result<PreparationTranscriptionModelSelection, ApplicationError> {
        select_preparation_model(&self.repository.list_models()?, language)
    }

    async fn ensure_recommended_preparation_model(
        self: &Arc<Self>,
        language: Option<&str>,
    ) -> Result<TranscriptionModelId, ApplicationError> {
        let _guard = self.recommended_model_install.lock().await;
        match self.recommended_preparation_model(language)? {
            PreparationTranscriptionModelSelection::Ready { model_id } => Ok(model_id),
            PreparationTranscriptionModelSelection::InstallRecommended { model_id } => {
                let model = self.clone().install_model(model_id.clone()).await?;
                if matches!(
                    model.state,
                    TranscriptionModelState::Installed | TranscriptionModelState::Custom
                ) {
                    Ok(model_id)
                } else {
                    Err(ApplicationError::Repository(model.error.unwrap_or_else(
                        || "recommended ASR model installation failed".into(),
                    )))
                }
            }
        }
    }

    fn build_job_candidate(
        &self,
        request: CreateJobRequest,
        retry_of_job_id: Option<TranscriptionJobId>,
        deterministic_id: Option<TranscriptionJobId>,
    ) -> Result<TranscriptionJob, ApplicationError> {
        let media_id = MediaId::parse(request.media_id)?;
        let media = self
            .services
            .media_analysis()
            .read_media(&media_id)?
            .ok_or(ApplicationError::NotFound("media"))?;
        let model_id = TranscriptionModelId::parse(request.model_id)?;
        let model = self
            .repository
            .get_model(&model_id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        if !matches!(
            model.state,
            TranscriptionModelState::Installed | TranscriptionModelState::Custom
        ) {
            return Err(ApplicationError::Validation(
                "installed transcription model",
            ));
        }
        let provider = self.providers().into_iter().next().expect("first provider");
        if !provider.available {
            return Err(ApplicationError::Validation("transcription runtime"));
        }
        if request.purpose == TranscriptionPurpose::TranslateToEnglish && model.english_only {
            return Err(ApplicationError::Validation(
                "multilingual translation model",
            ));
        }
        let settings_json = serde_json::json!({
            "destination": request.destination,
            "purpose": request.purpose,
            "language": request.language,
            "audio_track": request.audio_track
        })
        .to_string();
        let input_fingerprint = hex::encode(Sha256::digest(format!(
            "{}:{}:{}:{}",
            media.fingerprint,
            model.id.as_str(),
            model.checksum_sha256,
            settings_json
        )));
        let created_at_ms = now_ms();
        Ok(TranscriptionJob {
            id: deterministic_id.unwrap_or_else(|| {
                TranscriptionJobId::from_fingerprint(
                    "transcription-job",
                    &format!("{input_fingerprint}:{created_at_ms}"),
                )
            }),
            media_id,
            media_title: media.title,
            media_fingerprint: media.fingerprint,
            provider_id: provider.id,
            provider_version: provider.runtime_version.clone(),
            runtime_id: provider.runtime_id,
            runtime_version: provider.runtime_version,
            model_id: model.id,
            model_revision: model.revision,
            model_checksum_sha256: model.checksum_sha256,
            destination: request.destination,
            purpose: request.purpose,
            requested_language: request.language,
            detected_language: None,
            audio_track: request.audio_track,
            settings_json,
            input_fingerprint,
            status: TranscriptionJobStatus::Queued,
            phase_progress: 0,
            error_code: None,
            error_message: None,
            retry_of_job_id,
            generated_track_id: None,
            created_at_ms,
            started_at_ms: None,
            completed_at_ms: None,
            updated_at_ms: created_at_ms,
            archived_at_ms: None,
        })
    }

    fn claim_preparation_job(
        &self,
        candidate: TranscriptionJob,
        terminal_policy: PreparationTranscriptionTerminalPolicy,
    ) -> Result<(EnsuredPreparationTranscription, bool), ApplicationError> {
        let job = match self.repository.create_job_if_absent(&candidate)? {
            CreateTranscriptionJob::Created(job) => {
                self.emit(EventName::TranscriptionJobChanged, &job);
                return Ok((
                    EnsuredPreparationTranscription {
                        disposition: PreparationTranscriptionDisposition::Created,
                        job,
                    },
                    true,
                ));
            }
            CreateTranscriptionJob::Existing(job) => job,
        };
        if job.id != candidate.id
            || job.media_id != candidate.media_id
            || job.media_fingerprint != candidate.media_fingerprint
            || job.input_fingerprint != candidate.input_fingerprint
        {
            return Err(ApplicationError::Conflict(
                "preparation transcription idempotency key is already bound to different inputs",
            ));
        }
        self.claim_existing_preparation_job(job, terminal_policy)
    }

    fn claim_existing_preparation_job(
        &self,
        mut job: TranscriptionJob,
        terminal_policy: PreparationTranscriptionTerminalPolicy,
    ) -> Result<(EnsuredPreparationTranscription, bool), ApplicationError> {
        loop {
            let disposition = match job.status {
                TranscriptionJobStatus::Queued
                | TranscriptionJobStatus::Extracting
                | TranscriptionJobStatus::Transcribing
                | TranscriptionJobStatus::Importing => {
                    PreparationTranscriptionDisposition::ExistingActive
                }
                TranscriptionJobStatus::Completed if self.completed_track_is_reusable(&job)? => {
                    PreparationTranscriptionDisposition::ReusedCompleted
                }
                TranscriptionJobStatus::Completed => {
                    let expected_status = job.status;
                    let restarted = reset_preparation_job(job.clone(), now_ms());
                    match self
                        .repository
                        .transition_job(expected_status, &restarted)?
                    {
                        TranscriptionJobTransition::Applied(restarted) => {
                            self.emit(EventName::TranscriptionJobChanged, &restarted);
                            return Ok((
                                EnsuredPreparationTranscription {
                                    disposition:
                                        PreparationTranscriptionDisposition::RestartedInvalidCompletion,
                                    job: restarted,
                                },
                                true,
                            ));
                        }
                        TranscriptionJobTransition::Rejected(current) => {
                            job = current;
                            continue;
                        }
                    }
                }
                TranscriptionJobStatus::Failed
                    if job.error_code.as_deref() == Some("interrupted") =>
                {
                    let expected_status = job.status;
                    let restarted = reset_preparation_job(job.clone(), now_ms());
                    match self
                        .repository
                        .transition_job(expected_status, &restarted)?
                    {
                        TranscriptionJobTransition::Applied(restarted) => {
                            self.emit(EventName::TranscriptionJobChanged, &restarted);
                            return Ok((
                                EnsuredPreparationTranscription {
                                    disposition:
                                        PreparationTranscriptionDisposition::RestartedInterrupted,
                                    job: restarted,
                                },
                                true,
                            ));
                        }
                        TranscriptionJobTransition::Rejected(current) => {
                            job = current;
                            continue;
                        }
                    }
                }
                TranscriptionJobStatus::Cancelled | TranscriptionJobStatus::Failed
                    if terminal_policy == PreparationTranscriptionTerminalPolicy::Restart =>
                {
                    let expected_status = job.status;
                    let restarted = reset_preparation_job(job.clone(), now_ms());
                    match self
                        .repository
                        .transition_job(expected_status, &restarted)?
                    {
                        TranscriptionJobTransition::Applied(restarted) => {
                            self.emit(EventName::TranscriptionJobChanged, &restarted);
                            return Ok((
                                EnsuredPreparationTranscription {
                                    disposition:
                                        PreparationTranscriptionDisposition::RestartedExplicitRetry,
                                    job: restarted,
                                },
                                true,
                            ));
                        }
                        TranscriptionJobTransition::Rejected(current) => {
                            job = current;
                            continue;
                        }
                    }
                }
                TranscriptionJobStatus::Cancelled | TranscriptionJobStatus::Failed => {
                    PreparationTranscriptionDisposition::ExistingTerminal
                }
            };
            return Ok((EnsuredPreparationTranscription { disposition, job }, false));
        }
    }

    fn completed_track_is_reusable(
        &self,
        job: &TranscriptionJob,
    ) -> Result<bool, ApplicationError> {
        let Some(track_id) = job.generated_track_id.as_ref() else {
            return Ok(false);
        };
        let analysis = self.services.media_analysis();
        let Some(media) = analysis.read_media(&job.media_id)? else {
            return Ok(false);
        };
        if media.fingerprint != job.media_fingerprint
            || media.availability != MediaAvailability::Available
        {
            return Ok(false);
        }
        Ok(analysis
            .read_subtitle_track(track_id)?
            .is_some_and(|track| {
                track.media_id == job.media_id && track.status == SubtitleTrackStatus::Available
            }))
    }

    fn start_job(self: Arc<Self>, id: TranscriptionJobId) {
        tokio::spawn(async move { self.run_job(id).await });
    }

    pub fn cancel_job(
        &self,
        id: &TranscriptionJobId,
    ) -> Result<TranscriptionJob, ApplicationError> {
        let mut job = self
            .repository
            .get_job(id)?
            .ok_or(ApplicationError::NotFound("transcription job"))?;
        loop {
            if !matches!(
                job.status,
                TranscriptionJobStatus::Queued
                    | TranscriptionJobStatus::Extracting
                    | TranscriptionJobStatus::Transcribing
            ) {
                return Ok(job);
            }
            let mut cancelled = job.clone();
            let cancelled_at_ms = now_ms();
            cancelled.status = TranscriptionJobStatus::Cancelled;
            cancelled.completed_at_ms = Some(cancelled_at_ms);
            cancelled.updated_at_ms = cancelled_at_ms;
            match self.repository.transition_job(job.status, &cancelled)? {
                TranscriptionJobTransition::Applied(job) => {
                    self.emit(EventName::TranscriptionJobChanged, &job);
                    return Ok(job);
                }
                TranscriptionJobTransition::Rejected(current) => job = current,
            }
        }
    }

    pub fn retry_job(
        self: Arc<Self>,
        id: &TranscriptionJobId,
    ) -> Result<TranscriptionJob, ApplicationError> {
        let old = self
            .repository
            .get_job(id)?
            .ok_or(ApplicationError::NotFound("transcription job"))?;
        self.clone().create_job_with_retry(
            CreateJobRequest {
                media_id: old.media_id.as_str().into(),
                model_id: old.model_id.as_str().into(),
                destination: old.destination,
                purpose: old.purpose,
                language: old.requested_language,
                audio_track: old.audio_track,
                force: true,
            },
            Some(old.id),
        )
    }

    pub fn archive_job(
        &self,
        id: &TranscriptionJobId,
    ) -> Result<TranscriptionJob, ApplicationError> {
        let mut job = self
            .repository
            .get_job(id)?
            .ok_or(ApplicationError::NotFound("transcription job"))?;
        if matches!(
            job.status,
            TranscriptionJobStatus::Queued
                | TranscriptionJobStatus::Extracting
                | TranscriptionJobStatus::Transcribing
                | TranscriptionJobStatus::Importing
        ) {
            return Err(ApplicationError::Validation(
                "active transcription job cannot be archived",
            ));
        }
        let expected_status = job.status;
        job.archived_at_ms = Some(now_ms());
        job.updated_at_ms = now_ms();
        match self.repository.transition_job(expected_status, &job)? {
            TranscriptionJobTransition::Applied(job) => {
                self.emit(EventName::TranscriptionJobChanged, &job);
                Ok(job)
            }
            TranscriptionJobTransition::Rejected(current) => Ok(current),
        }
    }

    async fn run_job(self: Arc<Self>, id: TranscriptionJobId) {
        let _permit = match self.queue.acquire().await {
            Ok(value) => value,
            Err(_) => return,
        };
        let result = self.execute_job(&id).await;
        let _ = tokio::fs::remove_dir_all(self.temp_dir.join(id.as_str())).await;
        if let Err(error) = result
            && let Ok(Some(job)) = self.repository.get_job(&id)
            && matches!(
                job.status,
                TranscriptionJobStatus::Queued
                    | TranscriptionJobStatus::Extracting
                    | TranscriptionJobStatus::Transcribing
                    | TranscriptionJobStatus::Importing
            )
        {
            let expected_status = job.status;
            let mut failed = job;
            failed.status = TranscriptionJobStatus::Failed;
            failed.error_code = Some("transcription_failed".into());
            failed.error_message = Some(error.to_string());
            failed.completed_at_ms = Some(now_ms());
            failed.updated_at_ms = now_ms();
            if let Ok(TranscriptionJobTransition::Applied(failed)) =
                self.repository.transition_job(expected_status, &failed)
            {
                self.emit(EventName::TranscriptionJobChanged, &failed);
            }
        }
    }

    async fn run_recording_transcription(self: Arc<Self>, id: RecordingTranscriptionJobId) {
        let _permit = match self.queue.acquire().await {
            Ok(value) => value,
            Err(_) => return,
        };
        let result = self.execute_recording_transcription(&id).await;
        let _ = tokio::fs::remove_dir_all(self.temp_dir.join("recordings").join(id.as_str())).await;
        if let Err(error) = result {
            let mut jobs = self
                .recording_jobs
                .lock()
                .expect("recording transcription jobs mutex poisoned");
            if let Some(job) = jobs.get_mut(&id)
                && job.status != RecordingTranscriptionStatus::Cancelled
            {
                let completed_at_ms = now_ms();
                job.status = RecordingTranscriptionStatus::Failed;
                job.error_code = Some("recording_transcription_failed".into());
                job.error_message = Some(error.to_string());
                job.completed_at_ms = Some(completed_at_ms);
                job.latency_ms = Some(completed_at_ms.saturating_sub(job.created_at_ms));
            }
        }
    }

    async fn execute_recording_transcription(
        &self,
        id: &RecordingTranscriptionJobId,
    ) -> Result<(), ApplicationError> {
        let mut job = self
            .recording_transcription_job(id)
            .ok_or(ApplicationError::NotFound("recording transcription job"))?;
        if job.status == RecordingTranscriptionStatus::Cancelled {
            return Ok(());
        }
        let recording = self
            .services
            .recordings()
            .recording_asset(&job.recording_asset_id)?
            .ok_or(ApplicationError::NotFound("recording asset"))?;
        validate_recording_input(&recording)?;
        if hash_file(Path::new(&recording.file_path))? != recording.audio.content_sha256 {
            return Err(ApplicationError::Validation("recording file integrity"));
        }
        let model = self
            .repository
            .get_model(&job.provenance.model_id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        let model_path = model
            .local_path
            .ok_or(ApplicationError::Validation("model path"))?;
        let whisper = resolve_tool("LLPLAYERNEXT_WHISPER_CLI", "whisper-cli")
            .ok_or(ApplicationError::Validation("whisper runtime"))?;
        let work = self.temp_dir.join("recordings").join(id.as_str());
        tokio::fs::create_dir_all(&work).await.map_err(io_error)?;
        let output = work.join("result");
        let started_at_ms = now_ms();
        job.status = RecordingTranscriptionStatus::Transcribing;
        job.started_at_ms = Some(started_at_ms);
        self.update_recording_job(job);
        let args = vec![
            "-m".into(),
            model_path,
            "-f".into(),
            recording.file_path,
            // Standard JSON carries segment offsets/text without the token
            // dump. whisper.cpp full JSON can contain split non-ASCII token
            // bytes (not valid UTF-8 JSON) for Mandarin recordings.
            "-oj".into(),
            "-of".into(),
            output.to_string_lossy().into_owned(),
            "-l".into(),
            self.recording_transcription_job(id)
                .and_then(|value| value.provenance.requested_language)
                .unwrap_or_else(|| "auto".into()),
        ];
        self.process_runner
            .run(
                ProcessSpec::new(whisper, args),
                Arc::new(RecordingTranscriptionCancellation {
                    jobs: self.recording_jobs.clone(),
                    job_id: id.clone(),
                }),
            )
            .await?;
        let bytes = tokio::fs::read(output.with_extension("json"))
            .await
            .map_err(io_error)?;
        let result = parse_recording_transcription_json(&bytes)?;
        let completed_at_ms = now_ms();
        let mut job = self
            .recording_transcription_job(id)
            .ok_or(ApplicationError::NotFound("recording transcription job"))?;
        if job.status == RecordingTranscriptionStatus::Cancelled {
            return Ok(());
        }
        job.status = RecordingTranscriptionStatus::Completed;
        job.raw_transcript = Some(
            result
                .segments
                .iter()
                .map(|segment| segment.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        );
        job.segments = result.segments;
        job.provenance.detected_language = result.detected_language;
        job.completed_at_ms = Some(completed_at_ms);
        job.latency_ms = Some(completed_at_ms.saturating_sub(started_at_ms));
        self.update_recording_job(job);
        Ok(())
    }

    fn update_recording_job(&self, job: RecordingTranscriptionJob) {
        self.recording_jobs
            .lock()
            .expect("recording transcription jobs mutex poisoned")
            .insert(job.id.clone(), job);
    }

    async fn execute_job(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError> {
        let mut job = self
            .repository
            .get_job(id)?
            .ok_or(ApplicationError::NotFound("transcription job"))?;
        if job.status != TranscriptionJobStatus::Queued {
            return Ok(());
        }
        job.started_at_ms = Some(now_ms());
        if !self.transition(&mut job, TranscriptionJobStatus::Extracting, 5)? {
            return Ok(());
        }
        let media = self
            .services
            .media_analysis()
            .read_media(&job.media_id)?
            .ok_or(ApplicationError::NotFound("media"))?;
        let model = self
            .repository
            .get_model(&job.model_id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        let dtw_preset = dtw_preset_for_model(&model);
        let model_path = model
            .local_path
            .ok_or(ApplicationError::Validation("model path"))?;
        let ffmpeg = resolve_tool("LLPLAYERNEXT_FFMPEG", "ffmpeg")
            .ok_or(ApplicationError::Validation("ffmpeg runtime"))?;
        let whisper = resolve_tool("LLPLAYERNEXT_WHISPER_CLI", "whisper-cli")
            .ok_or(ApplicationError::Validation("whisper runtime"))?;
        let work = self.temp_dir.join(id.as_str());
        tokio::fs::create_dir_all(&work).await.map_err(io_error)?;
        let wav = work.join("audio.wav");
        let ffmpeg_args = ffmpeg_wav_args(media.path, job.audio_track, &wav);
        self.run_command(&job.id, &ffmpeg, &ffmpeg_args).await?;
        if !self.transition(&mut job, TranscriptionJobStatus::Transcribing, 35)? {
            return Ok(());
        }
        let output = work.join("result");
        let mut whisper_args = vec![
            "-m".into(),
            model_path.clone(),
            "-f".into(),
            wav.to_string_lossy().into_owned(),
            "-osrt".into(),
            "-ojf".into(),
            "-of".into(),
            output.to_string_lossy().into_owned(),
        ];
        if let Some(preset) = dtw_preset.as_ref() {
            whisper_args.push("-dtw".into());
            whisper_args.push(preset.clone());
        }
        whisper_args.extend([
            "-l".into(),
            job.requested_language
                .clone()
                .unwrap_or_else(|| "auto".into()),
        ]);
        if job.purpose == TranscriptionPurpose::TranslateToEnglish {
            whisper_args.push("-tr".into());
        }
        self.run_command(&job.id, &whisper, &whisper_args).await?;
        job.detected_language = tokio::fs::read(output.with_extension("json"))
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|value| {
                value
                    .pointer("/result/language")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .or_else(|| job.requested_language.clone());
        // Entering Importing is the irreversible commit point. Cancellation is
        // accepted only before this CAS succeeds, so a durable Cancelled state
        // proves that subtitle import never began.
        if !self.transition(&mut job, TranscriptionJobStatus::Importing, 90)? {
            return Ok(());
        }
        let srt = tokio::fs::read(output.with_extension("srt"))
            .await
            .map_err(io_error)?;
        let track = self
            .services
            .media_analysis()
            .import_subtitle(ImportSubtitle {
                media_id: job.media_id.clone(),
                source_name: format!("ASR-{}.srt", file_id(&model.display_name)),
                content: srt,
                language: resolved_subtitle_language(&job),
                identity_salt: Some(format!(
                    "{}:{}:{}",
                    job.provider_id,
                    job.model_id.as_str(),
                    job.model_revision
                )),
            })?;
        // Store the text-line word timeline from JSON-full output when DTW was
        // enabled. The sound line is a separate, independently triggered workflow
        // (SoundLineCoordinator, driven off the transcription-completed event) —
        // no sound-line work happens on the transcription path.
        if dtw_preset.is_some() {
            let json_path = output.with_extension("json");
            if let Ok(json_bytes) = tokio::fs::read(&json_path).await
                && let Ok(Some(result)) = self
                    .services
                    .media_analysis()
                    .store_transcription_text_word_timeline(&track.id, &json_bytes)
                    .await
            {
                let _ = self.events.send(
                    crate::events::WordTimingsCompletedPayload {
                        job_id: None,
                        track_id: track.id.as_str().to_owned(),
                        line: Some("text".to_owned()),
                        count: result.extracted_word_count,
                        timeline_id: result
                            .final_timeline_id
                            .as_ref()
                            .map(|id| id.as_str().to_owned()),
                    }
                    .envelope(),
                );
            }
        }
        self.repository.save_provenance(&SubtitleTrackProvenance {
            track_id: track.id.clone(),
            transcription_job_id: job.id.clone(),
            provider_id: job.provider_id.clone(),
            runtime_version: job.runtime_version.clone(),
            model_id: job.model_id.clone(),
            model_revision: job.model_revision.clone(),
            model_checksum_sha256: job.model_checksum_sha256.clone(),
            settings_json: job.settings_json.clone(),
            created_at_ms: now_ms(),
        })?;
        job.generated_track_id = Some(track.id);
        if !self.transition(&mut job, TranscriptionJobStatus::Completed, 100)? {
            return Ok(());
        }
        let _ = tokio::fs::remove_dir_all(work).await;
        Ok(())
    }

    async fn run_command(
        &self,
        job_id: &TranscriptionJobId,
        executable: &Path,
        args: &[String],
    ) -> Result<(), ApplicationError> {
        self.process_runner
            .run(
                ProcessSpec::new(executable, args.to_vec()),
                Arc::new(TranscriptionCancellation {
                    repository: self.repository.clone(),
                    job_id: job_id.clone(),
                }),
            )
            .await
    }

    fn transition(
        &self,
        job: &mut TranscriptionJob,
        status: TranscriptionJobStatus,
        progress: u8,
    ) -> Result<bool, ApplicationError> {
        let expected_status = job.status;
        let mut candidate = job.clone();
        candidate.status = status;
        candidate.phase_progress = progress;
        candidate.updated_at_ms = now_ms();
        if status == TranscriptionJobStatus::Completed {
            candidate.completed_at_ms = Some(candidate.updated_at_ms);
        }
        match self
            .repository
            .transition_job(expected_status, &candidate)?
        {
            TranscriptionJobTransition::Applied(updated) => {
                *job = updated;
                self.emit(EventName::TranscriptionJobChanged, job);
                Ok(true)
            }
            TranscriptionJobTransition::Rejected(current) => {
                *job = current;
                Ok(false)
            }
        }
    }

    fn emit<T: serde::Serialize>(&self, event: EventName, value: &T) {
        let _ = self.events.send(EventEnvelope::v1(
            event,
            serde_json::to_value(value).expect("transcription event serializes"),
        ));
    }

    fn seed_catalog(&self) -> Result<(), ApplicationError> {
        let existing = self.repository.list_models()?;
        for model in catalog() {
            if existing.iter().any(|value| value.id == model.id) {
                continue;
            }
            self.repository.upsert_model(&model)?;
        }
        Ok(())
    }
}

struct TranscriptionCancellation {
    repository: Arc<dyn TranscriptionRepository>,
    job_id: TranscriptionJobId,
}

struct RecordingTranscriptionCancellation {
    jobs: Arc<Mutex<HashMap<RecordingTranscriptionJobId, RecordingTranscriptionJob>>>,
    job_id: RecordingTranscriptionJobId,
}

struct TranscriptionDownloadProgress {
    repository: Arc<dyn TranscriptionRepository>,
    events: broadcast::Sender<EventEnvelope>,
    model_id: TranscriptionModelId,
}

#[derive(Default)]
struct PreparationAudioTrackCounter {
    count: AtomicUsize,
}

impl PreparationAudioTrackCounter {
    fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

impl ProcessOutputObserver for PreparationAudioTrackCounter {
    fn stdout_line(&self, line: &str) -> Result<(), ApplicationError> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }
        line.parse::<u32>().map_err(|_| {
            ApplicationError::ExternalProcess(
                "ffprobe returned an invalid audio stream index".into(),
            )
        })?;
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl DownloadProgress for TranscriptionDownloadProgress {
    fn is_cancelled(&self) -> Result<bool, ApplicationError> {
        Ok(self
            .repository
            .get_model(&self.model_id)?
            .is_some_and(|model| model.state != TranscriptionModelState::Installing))
    }

    fn downloaded(&self, bytes: u64) -> Result<(), ApplicationError> {
        let mut model = self
            .repository
            .get_model(&self.model_id)?
            .ok_or(ApplicationError::NotFound("transcription model"))?;
        model.installed_bytes = bytes;
        model.updated_at_ms = now_ms();
        let model = self.repository.upsert_model(&model)?;
        let _ = self.events.send(EventEnvelope::v1(
            EventName::TranscriptionModelChanged,
            serde_json::to_value(model).expect("transcription model serializes"),
        ));
        Ok(())
    }
}

impl CancellationProbe for TranscriptionCancellation {
    fn is_cancelled(&self) -> Result<bool, ApplicationError> {
        Ok(self
            .repository
            .get_job(&self.job_id)?
            .is_some_and(|job| job.status == TranscriptionJobStatus::Cancelled))
    }
}

impl CancellationProbe for RecordingTranscriptionCancellation {
    fn is_cancelled(&self) -> Result<bool, ApplicationError> {
        Ok(self
            .jobs
            .lock()
            .expect("recording transcription jobs mutex poisoned")
            .get(&self.job_id)
            .is_some_and(|job| job.status == RecordingTranscriptionStatus::Cancelled))
    }
}

fn validate_recording_input(recording: &domain::RecordingAsset) -> Result<(), ApplicationError> {
    if recording.duration_ms > 120_000
        || recording.audio.container != "wav"
        || recording.audio.codec != "pcm_s16le"
        || recording.audio.sample_rate_hz != 16_000
        || recording.audio.channels != 1
        || recording.audio.sample_format != "s16"
    {
        return Err(ApplicationError::Validation(
            "16 kHz mono PCM s16 WAV recording",
        ));
    }
    let metadata = std::fs::metadata(&recording.file_path)
        .map_err(|_| ApplicationError::Validation("available recording file"))?;
    if metadata.len() != recording.audio.byte_length {
        return Err(ApplicationError::Validation("recording file byte length"));
    }
    Ok(())
}

fn parse_recording_transcription_json(
    bytes: &[u8],
) -> Result<domain::TranscriptionResult, ApplicationError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| ApplicationError::Repository(error.to_string()))?;
    let detected_language = value
        .pointer("/result/language")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let segments = value
        .get("transcription")
        .and_then(serde_json::Value::as_array)
        .ok_or(ApplicationError::Validation(
            "whisper.cpp recording transcription JSON",
        ))?
        .iter()
        .filter_map(|segment| {
            let offsets = segment.get("offsets")?;
            Some(TranscriptionSegment {
                start_ms: offsets.get("from")?.as_u64()?,
                end_ms: offsets.get("to")?.as_u64()?,
                text: segment.get("text")?.as_str()?.trim().to_owned(),
            })
        })
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return Err(ApplicationError::Validation(
            "non-empty recording transcript",
        ));
    }
    Ok(domain::TranscriptionResult {
        detected_language,
        segments,
    })
}

fn reset_preparation_job(mut job: TranscriptionJob, updated_at_ms: u64) -> TranscriptionJob {
    job.status = TranscriptionJobStatus::Queued;
    job.phase_progress = 0;
    job.detected_language = None;
    job.error_code = None;
    job.error_message = None;
    job.generated_track_id = None;
    job.started_at_ms = None;
    job.completed_at_ms = None;
    job.updated_at_ms = updated_at_ms;
    job.archived_at_ms = None;
    job
}

fn resolved_subtitle_language(job: &TranscriptionJob) -> Option<String> {
    if job.purpose == TranscriptionPurpose::TranslateToEnglish {
        Some("en".into())
    } else {
        job.requested_language
            .clone()
            .or_else(|| job.detected_language.clone())
    }
}

fn select_preparation_model(
    models: &[TranscriptionModelDescriptor],
    language: Option<&str>,
) -> Result<PreparationTranscriptionModelSelection, ApplicationError> {
    const RECOMMENDED_MODEL_ID: &str = "whisper.cpp:base@main";

    let mut installed = models
        .iter()
        .filter(|model| {
            matches!(
                model.state,
                TranscriptionModelState::Installed | TranscriptionModelState::Custom
            ) && preparation_model_supports_language(model, language)
        })
        .collect::<Vec<_>>();
    installed.sort_by_key(|model| {
        (
            model.id.as_str() != RECOMMENDED_MODEL_ID,
            transcription_quality_rank(model.quality),
            model.size_bytes,
            model.id.as_str(),
        )
    });
    if let Some(model) = installed.first() {
        return Ok(PreparationTranscriptionModelSelection::Ready {
            model_id: model.id.clone(),
        });
    }

    let recommended = models
        .iter()
        .find(|model| model.id.as_str() == RECOMMENDED_MODEL_ID)
        .ok_or(ApplicationError::NotFound(
            "recommended transcription model",
        ))?;
    if recommended.english_only {
        return Err(ApplicationError::Invalid(
            "recommended preparation transcription model must be multilingual".into(),
        ));
    }
    Ok(PreparationTranscriptionModelSelection::InstallRecommended {
        model_id: recommended.id.clone(),
    })
}

fn preparation_model_supports_language(
    model: &TranscriptionModelDescriptor,
    language: Option<&str>,
) -> bool {
    if !model.english_only {
        return true;
    }
    language.is_some_and(|language| {
        let language = language.trim().to_ascii_lowercase();
        language == "en" || language.starts_with("en-")
    })
}

fn transcription_quality_rank(quality: TranscriptionQuality) -> u8 {
    match quality {
        TranscriptionQuality::Fast => 0,
        TranscriptionQuality::Balanced => 1,
        TranscriptionQuality::Accurate => 2,
    }
}

fn catalog() -> Vec<TranscriptionModelDescriptor> {
    [
        (
            "base.en",
            147964211,
            "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
            TranscriptionQuality::Fast,
            true,
        ),
        (
            "base",
            147951465,
            "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
            TranscriptionQuality::Fast,
            false,
        ),
        (
            "small.en",
            487614201,
            "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
            TranscriptionQuality::Balanced,
            true,
        ),
        (
            "small",
            487601967,
            "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
            TranscriptionQuality::Balanced,
            false,
        ),
        (
            "medium.en",
            1533774781,
            "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
            TranscriptionQuality::Accurate,
            true,
        ),
        (
            "medium",
            1533763059,
            "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
            TranscriptionQuality::Accurate,
            false,
        ),
    ]
    .into_iter()
    .map(
        |(name, size, checksum, quality, english_only)| TranscriptionModelDescriptor {
            id: TranscriptionModelId::parse(format!("whisper.cpp:{name}@main"))
                .expect("catalog ID"),
            provider_id: "whisper.cpp".into(),
            display_name: name.into(),
            family: "whisper".into(),
            revision: "main-80da2d8".into(),
            checksum_sha256: checksum.into(),
            download_url: Some(format!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
            )),
            local_path: None,
            size_bytes: size,
            quality,
            english_only,
            supports_translation: !english_only,
            state: TranscriptionModelState::Downloadable,
            installed_bytes: 0,
            error: None,
            license: "MIT".into(),
            updated_at_ms: now_ms(),
        },
    )
    .collect()
}

fn dtw_preset_for_model(model: &TranscriptionModelDescriptor) -> Option<String> {
    if model.provider_id != "whisper.cpp" {
        return None;
    }
    let mut candidates = vec![model.display_name.as_str()];
    if let Some(path) = model.local_path.as_deref()
        && let Some(name) = Path::new(path).file_name().and_then(|value| value.to_str())
    {
        candidates.push(name);
    }
    candidates.into_iter().find_map(dtw_preset_from_name)
}

fn dtw_preset_from_name(name: &str) -> Option<String> {
    let mut normalized = name.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if let Some(file_name) = Path::new(&normalized)
        .file_name()
        .and_then(|value| value.to_str())
    {
        normalized = file_name.to_string();
    }
    if let Some(stripped) = normalized.strip_suffix(".bin") {
        normalized = stripped.to_string();
    }
    normalized = normalized.replace('_', "-");
    for prefix in ["ggml-", "whisper-"] {
        if let Some(stripped) = normalized.strip_prefix(prefix) {
            normalized = stripped.to_string();
        }
    }
    if let Some(index) = normalized.find("-q") {
        normalized.truncate(index);
    }
    if let Some(index) = normalized.find(".q") {
        normalized.truncate(index);
    }
    let preset = match normalized.as_str() {
        "tiny" | "tiny.en" | "base" | "base.en" | "small" | "small.en" | "medium" | "medium.en" => {
            normalized
        }
        "large-v1" | "large.v1" => "large.v1".into(),
        "large-v2" | "large.v2" => "large.v2".into(),
        "large-v3" | "large.v3" => "large.v3".into(),
        "large-v3-turbo" | "large.v3.turbo" => "large.v3.turbo".into(),
        _ => return None,
    };
    Some(preset)
}

fn english_only_model_name(name: &str) -> bool {
    if dtw_preset_from_name(name).is_some_and(|preset| preset.ends_with(".en")) {
        return true;
    }
    let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
    ["tiny", "base", "small", "medium"]
        .iter()
        .any(|family| normalized.contains(&format!("-{family}-en-")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeProcessRunner;
    use crate::runtime_support::resolve_bundled_tool;
    use application::{AppServices, ImportSubtitle, RegisterMedia};
    use domain::MediaKind;
    use persistence_sqlite::SqliteRepository;

    fn coordinator_fixture(
        process_runner: Arc<dyn ProcessRunner>,
        ffprobe: Option<PathBuf>,
    ) -> (
        Arc<TranscriptionCoordinator>,
        Arc<SqliteRepository>,
        AppServices,
    ) {
        let repository = Arc::new(SqliteRepository::in_memory().unwrap());
        let services = AppServices::new(
            repository.clone(),
            repository.clone(),
            repository.clone(),
            repository.clone(),
            repository.clone(),
            repository.clone(),
            repository.clone(),
            repository.clone(),
        );
        let (events, _) = broadcast::channel(32);
        let coordinator = Arc::new(
            TranscriptionCoordinator::new_with_adapters_and_ffprobe(
                services.clone(),
                repository.clone(),
                events,
                process_runner,
                Arc::new(crate::FakeArtifactDownloader::new(Vec::new())),
                ffprobe,
            )
            .unwrap(),
        );
        (coordinator, repository, services)
    }

    fn media_job(
        id: TranscriptionJobId,
        media_id: MediaId,
        media_fingerprint: &str,
        status: TranscriptionJobStatus,
    ) -> TranscriptionJob {
        TranscriptionJob {
            id,
            media_id,
            media_title: "Preparation media".into(),
            media_fingerprint: media_fingerprint.into(),
            provider_id: "whisper.cpp".into(),
            provider_version: "test".into(),
            runtime_id: "whisper.cpp-cli".into(),
            runtime_version: "test".into(),
            model_id: TranscriptionModelId::parse("whisper.cpp:base@main").unwrap(),
            model_revision: "main-80da2d8".into(),
            model_checksum_sha256: "checksum".into(),
            destination: TranscriptionDestination::Primary,
            purpose: TranscriptionPurpose::Transcribe,
            requested_language: Some("en".into()),
            detected_language: Some("en".into()),
            audio_track: Some(0),
            settings_json:
                r#"{"destination":"primary","purpose":"transcribe","language":"en","audio_track":0}"#
                    .into(),
            input_fingerprint: "preparation-input".into(),
            status,
            phase_progress: if status == TranscriptionJobStatus::Completed {
                100
            } else {
                35
            },
            error_code: None,
            error_message: None,
            retry_of_job_id: None,
            generated_track_id: None,
            created_at_ms: 1,
            started_at_ms: Some(2),
            completed_at_ms: (status == TranscriptionJobStatus::Completed).then_some(3),
            updated_at_ms: 3,
            archived_at_ms: None,
        }
    }

    #[test]
    fn preparation_child_identity_is_deterministic_and_audio_selection_is_typed() {
        let first = preparation_transcription_child_id("media:audio:policy").unwrap();
        let second = preparation_transcription_child_id("media:audio:policy").unwrap();
        assert_eq!(first, second);
        assert!(preparation_transcription_child_id(" ").is_err());
        assert_eq!(
            resolve_preparation_audio_selection(0, None),
            PreparationAudioSelection::Unavailable {
                reason: "audio_track_not_found"
            }
        );
        assert_eq!(
            resolve_preparation_audio_selection(2, None),
            PreparationAudioSelection::SelectionRequired {
                reason: "audio_track_required"
            }
        );
        assert_eq!(
            resolve_preparation_audio_selection(1, None),
            PreparationAudioSelection::Selected { audio_track: 0 }
        );
        assert_eq!(
            resolve_preparation_audio_selection(1, Some(2)),
            PreparationAudioSelection::Unavailable {
                reason: "audio_track_not_found"
            }
        );
    }

    #[tokio::test]
    async fn preparation_audio_probe_reports_multitrack_selection_without_ffmpeg_defaulting() {
        let runner = Arc::new(FakeProcessRunner::with_stdout_lines([
            "0".into(),
            "1".into(),
        ]));
        let (coordinator, _repository, services) =
            coordinator_fixture(runner.clone(), Some(PathBuf::from("/test/ffprobe")));
        let media = services
            .media_analysis()
            .register_media(RegisterMedia {
                path: "/test/media.mkv".into(),
                fingerprint: "audio-probe-media".into(),
                title: "Audio probe".into(),
                kind: MediaKind::Video,
                duration_ms: Some(1_000),
            })
            .unwrap();

        assert_eq!(
            coordinator
                .resolve_preparation_audio_track(&media.id, None)
                .await
                .unwrap(),
            PreparationAudioSelection::SelectionRequired {
                reason: "audio_track_required"
            }
        );
        assert_eq!(
            coordinator
                .resolve_preparation_audio_track(&media.id, Some(1))
                .await
                .unwrap(),
            PreparationAudioSelection::Selected { audio_track: 1 }
        );
        assert_eq!(runner.calls()[0].executable, PathBuf::from("/test/ffprobe"));
    }

    #[test]
    fn preparation_claim_reuses_valid_completion_and_restarts_invalid_or_interrupted_child() {
        let (coordinator, repository, services) =
            coordinator_fixture(Arc::new(FakeProcessRunner::succeeding()), None);
        let media = services
            .media_analysis()
            .register_media(RegisterMedia {
                path: "/test/media.mkv".into(),
                fingerprint: "preparation-media".into(),
                title: "Preparation".into(),
                kind: MediaKind::Video,
                duration_ms: Some(1_000),
            })
            .unwrap();
        let track = services
            .media_analysis()
            .import_subtitle(ImportSubtitle {
                media_id: media.id.clone(),
                source_name: "asr.srt".into(),
                content: b"1\n00:00:00,000 --> 00:00:00,900\nHello.\n".to_vec(),
                language: Some("en".into()),
                identity_salt: Some("preparation-input".into()),
            })
            .unwrap();

        let completed_id = preparation_transcription_child_id("completed-child").unwrap();
        let mut completed = media_job(
            completed_id,
            media.id.clone(),
            &media.fingerprint,
            TranscriptionJobStatus::Completed,
        );
        completed.generated_track_id = Some(track.id.clone());
        repository.create_job(&completed).unwrap();
        let candidate = completed.clone();

        let (reused, should_start) = coordinator
            .claim_preparation_job(
                candidate.clone(),
                PreparationTranscriptionTerminalPolicy::Preserve,
            )
            .unwrap();
        assert_eq!(
            reused.disposition,
            PreparationTranscriptionDisposition::ReusedCompleted
        );
        assert!(!should_start);

        services
            .media_analysis()
            .archive_subtitle_track(&track.id)
            .unwrap();
        let (restarted, should_start) = coordinator
            .claim_preparation_job(candidate, PreparationTranscriptionTerminalPolicy::Preserve)
            .unwrap();
        assert_eq!(
            restarted.disposition,
            PreparationTranscriptionDisposition::RestartedInvalidCompletion
        );
        assert_eq!(restarted.job.id, completed.id);
        assert_eq!(restarted.job.status, TranscriptionJobStatus::Queued);
        assert!(restarted.job.generated_track_id.is_none());
        assert!(should_start);

        let interrupted_id = preparation_transcription_child_id("interrupted-child").unwrap();
        let mut interrupted = media_job(
            interrupted_id,
            media.id.clone(),
            &media.fingerprint,
            TranscriptionJobStatus::Failed,
        );
        interrupted.error_code = Some("interrupted".into());
        interrupted.error_message = Some("service stopped".into());
        repository.create_job(&interrupted).unwrap();
        let (restarted, should_start) = coordinator
            .claim_preparation_job(
                interrupted.clone(),
                PreparationTranscriptionTerminalPolicy::Preserve,
            )
            .unwrap();
        assert_eq!(
            restarted.disposition,
            PreparationTranscriptionDisposition::RestartedInterrupted
        );
        assert_eq!(restarted.job.id, interrupted.id);
        assert_eq!(restarted.job.status, TranscriptionJobStatus::Queued);
        assert!(restarted.job.error_code.is_none());
        assert!(should_start);

        let failed_id = preparation_transcription_child_id("failed-child").unwrap();
        let mut failed = media_job(
            failed_id,
            media.id,
            &media.fingerprint,
            TranscriptionJobStatus::Failed,
        );
        failed.error_code = Some("provider_failed".into());
        repository.create_job(&failed).unwrap();
        let job_count = repository.list_jobs().unwrap().len();

        let (preserved, should_start) = coordinator
            .claim_preparation_job(
                failed.clone(),
                PreparationTranscriptionTerminalPolicy::Preserve,
            )
            .unwrap();
        assert_eq!(
            preserved.disposition,
            PreparationTranscriptionDisposition::ExistingTerminal
        );
        assert!(!should_start);

        let (restarted, should_start) = coordinator
            .claim_preparation_job(
                failed.clone(),
                PreparationTranscriptionTerminalPolicy::Restart,
            )
            .unwrap();
        assert_eq!(
            restarted.disposition,
            PreparationTranscriptionDisposition::RestartedExplicitRetry
        );
        assert_eq!(restarted.job.id, failed.id);
        assert_eq!(restarted.job.status, TranscriptionJobStatus::Queued);
        assert!(should_start);
        assert_eq!(repository.list_jobs().unwrap().len(), job_count);
    }

    #[tokio::test]
    async fn preparation_ensure_restarts_existing_child_without_reselecting_model() {
        let (coordinator, repository, services) =
            coordinator_fixture(Arc::new(FakeProcessRunner::succeeding()), None);
        let media = services
            .media_analysis()
            .register_media(RegisterMedia {
                path: "/test/media.mkv".into(),
                fingerprint: "existing-child-media".into(),
                title: "Preparation".into(),
                kind: MediaKind::Video,
                duration_ms: Some(1_000),
            })
            .unwrap();
        let idempotency_key = "existing-interrupted-child";
        let child_id = preparation_transcription_child_id(idempotency_key).unwrap();
        let mut interrupted = media_job(
            child_id.clone(),
            media.id.clone(),
            &media.fingerprint,
            TranscriptionJobStatus::Failed,
        );
        interrupted.error_code = Some("interrupted".into());
        interrupted.error_message = Some("service stopped".into());
        repository.create_job(&interrupted).unwrap();

        // Change the current automatic policy after the child was created.
        // Recovery must retain the child's persisted model and provenance.
        let mut old_model = repository
            .get_model(&interrupted.model_id)
            .unwrap()
            .unwrap();
        old_model.state = TranscriptionModelState::Downloadable;
        repository.upsert_model(&old_model).unwrap();
        let preferred_id = TranscriptionModelId::parse("whisper.cpp:small.en@main").unwrap();
        let mut newly_preferred = repository.get_model(&preferred_id).unwrap().unwrap();
        newly_preferred.state = TranscriptionModelState::Installed;
        repository.upsert_model(&newly_preferred).unwrap();

        let ensured = coordinator
            .ensure_preparation_transcription(EnsurePreparationTranscriptionRequest {
                idempotency_key: idempotency_key.into(),
                child_id,
                media_id: media.id,
                language: Some("en".into()),
                audio_track: 0,
                terminal_policy: PreparationTranscriptionTerminalPolicy::Preserve,
            })
            .await
            .unwrap();

        assert_eq!(
            ensured.disposition,
            PreparationTranscriptionDisposition::RestartedInterrupted
        );
        assert_eq!(ensured.job.id, interrupted.id);
        assert_eq!(ensured.job.model_id, interrupted.model_id);
        assert_eq!(ensured.job.input_fingerprint, interrupted.input_fingerprint);
        assert_eq!(ensured.job.status, TranscriptionJobStatus::Queued);
    }

    #[test]
    fn preparation_model_policy_prefers_compatible_installed_then_multilingual_base() {
        let mut models = catalog();
        let english = models
            .iter_mut()
            .find(|model| model.id.as_str() == "whisper.cpp:small.en@main")
            .unwrap();
        english.state = TranscriptionModelState::Installed;
        let english_id = english.id.clone();
        assert_eq!(
            select_preparation_model(&models, Some("en")).unwrap(),
            PreparationTranscriptionModelSelection::Ready {
                model_id: english_id
            }
        );
        assert_eq!(
            select_preparation_model(&models, Some("zh")).unwrap(),
            PreparationTranscriptionModelSelection::InstallRecommended {
                model_id: TranscriptionModelId::parse("whisper.cpp:base@main").unwrap()
            }
        );

        let multilingual = models
            .iter_mut()
            .find(|model| model.id.as_str() == "whisper.cpp:small@main")
            .unwrap();
        multilingual.state = TranscriptionModelState::Installed;
        let multilingual_id = multilingual.id.clone();
        assert_eq!(
            select_preparation_model(&models, Some("zh")).unwrap(),
            PreparationTranscriptionModelSelection::Ready {
                model_id: multilingual_id
            }
        );
    }

    #[test]
    fn subtitle_import_uses_detected_language_when_asr_language_was_automatic() {
        let mut job = media_job(
            TranscriptionJobId::parse("automatic-language").unwrap(),
            MediaId::parse("media").unwrap(),
            "media-fingerprint",
            TranscriptionJobStatus::Transcribing,
        );
        job.requested_language = None;
        job.detected_language = Some("ja".into());

        assert_eq!(resolved_subtitle_language(&job).as_deref(), Some("ja"));

        job.purpose = TranscriptionPurpose::TranslateToEnglish;
        assert_eq!(resolved_subtitle_language(&job).as_deref(), Some("en"));
    }

    #[test]
    fn resolves_development_runtime_from_repository_root() {
        let root = std::env::temp_dir().join(format!(
            "llplayernext-runtime-test-{}",
            application::now_ms()
        ));
        let runtime = root.join("third_party/runtime/macos-arm64");
        std::fs::create_dir_all(&runtime).unwrap();
        let whisper = runtime.join("whisper-cli");
        std::fs::write(&whisper, b"#!/bin/sh\n").unwrap();

        let executable = root.join("target/debug/api-http");
        let resolved =
            resolve_bundled_tool("whisper-cli", &executable, &root.join("apps/desktop")).unwrap();
        assert_eq!(resolved, whisper);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolves_dtw_presets_for_whisper_cpp_models() {
        assert_eq!(dtw_preset_from_name("base.en").as_deref(), Some("base.en"));
        assert_eq!(
            dtw_preset_from_name("ggml-large-v3.bin").as_deref(),
            Some("large.v3")
        );
        assert_eq!(
            dtw_preset_from_name("ggml-small.en-q5_0.bin").as_deref(),
            Some("small.en")
        );
        assert_eq!(
            dtw_preset_from_name("ggml-large-v3-turbo-q8_0.bin").as_deref(),
            Some("large.v3.turbo")
        );
        assert!(dtw_preset_from_name("not-a-whisper-model.bin").is_none());
    }

    #[test]
    fn custom_whisper_cpp_models_enable_dtw_from_registered_path() {
        let model = TranscriptionModelDescriptor {
            id: TranscriptionModelId::parse("custom").unwrap(),
            provider_id: "whisper.cpp".into(),
            display_name: "custom.bin".into(),
            family: "custom".into(),
            revision: "local".into(),
            checksum_sha256: "checksum".into(),
            download_url: None,
            local_path: Some("/models/ggml-large-v3-q5_0.bin".into()),
            size_bytes: 1,
            quality: TranscriptionQuality::Balanced,
            english_only: false,
            supports_translation: true,
            state: TranscriptionModelState::Custom,
            installed_bytes: 1,
            error: None,
            license: "User supplied".into(),
            updated_at_ms: 1,
        };
        assert_eq!(dtw_preset_for_model(&model).as_deref(), Some("large.v3"));
    }

    #[test]
    fn recognizes_english_only_custom_model_names() {
        assert!(english_only_model_name("ggml-base.en.bin"));
        assert!(english_only_model_name("whisper-cpp-base-en-main.bin"));
        assert!(!english_only_model_name("whisper-cpp-base-main.bin"));
    }

    #[test]
    fn non_whisper_cpp_models_do_not_enable_dtw() {
        let model = TranscriptionModelDescriptor {
            id: TranscriptionModelId::parse("custom").unwrap(),
            provider_id: "other".into(),
            display_name: "ggml-base.en.bin".into(),
            family: "whisper".into(),
            revision: "local".into(),
            checksum_sha256: "checksum".into(),
            download_url: None,
            local_path: Some("/models/ggml-base.en.bin".into()),
            size_bytes: 1,
            quality: TranscriptionQuality::Balanced,
            english_only: false,
            supports_translation: true,
            state: TranscriptionModelState::Custom,
            installed_bytes: 1,
            error: None,
            license: "User supplied".into(),
            updated_at_ms: 1,
        };
        assert!(dtw_preset_for_model(&model).is_none());
    }

    #[test]
    fn parses_short_recording_json_with_language_and_offsets() {
        let result = parse_recording_transcription_json(
            r#"{
                "result": {"language": "zh"},
                "transcription": [
                    {"offsets": {"from": 120, "to": 840}, "text": " 你好 "},
                    {"offsets": {"from": 840, "to": 1600}, "text": "世界"}
                ]
            }"#
            .as_bytes(),
        )
        .unwrap();
        assert_eq!(result.detected_language.as_deref(), Some("zh"));
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].start_ms, 120);
        assert_eq!(result.segments[0].text, "你好");
        assert_eq!(result.segments[1].end_ms, 1600);
    }

    #[test]
    fn rejects_short_recording_json_without_readable_segments() {
        let error = parse_recording_transcription_json(
            br#"{"result":{"language":"en"},"transcription":[]}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-empty recording transcript"));
    }
}

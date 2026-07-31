use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use application::{
    AppServices, ApplicationError, FoundationPreparationChildRef, FoundationPreparationIntent,
    FoundationPreparationRequest, FoundationPreparationSlot, FoundationPreparationTarget,
    LearningPreparationRunId, LearningPreparationRunStatus, MediaAudioTrackIndex,
    MediaLearningPreparation, MediaLearningPreparationCommand, MediaLearningPreparationId,
    MediaLearningPreparationInspector, MediaLearningPreparationRepository,
    MediaLearningPreparationRequest, MediaLearningPreparationSelectionRequired,
    MediaLearningPreparationSourceInspection, MediaLearningPreparationStatus,
    MediaLearningPreparationTarget, MediaLearningPreparationUseCases, PrepareFoundationResult,
    PrepareMediaLearningResult, SubtitleTextTrackSlot, SubtitleTextTrackSnapshot,
    foundation_text_snapshot_fingerprint, now_ms,
};
use async_trait::async_trait;
use domain::{
    MediaAvailability, SubtitleTrack, SubtitleTrackStatus, TranscriptionJob, TranscriptionJobId,
    TranscriptionJobStatus,
};

use crate::transcription::{
    EnsurePreparationTranscriptionRequest, PreparationAudioSelection,
    PreparationTranscriptionTerminalPolicy, preparation_transcription_child_id,
};
use crate::{LearningPreparationCoordinator, TranscriptionCoordinator};

#[cfg(not(test))]
const POLL_INTERVAL: Duration = Duration::from_millis(250);
#[cfg(test)]
const POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Resolves only product-level subtitle choices. Audio probing remains async
/// and is therefore completed by `MediaLearningPreparationCoordinator` before
/// the durable parent run is created.
struct LocalMediaLearningPreparationInspector {
    services: AppServices,
}

impl LocalMediaLearningPreparationInspector {
    fn snapshot(
        &self,
        target: &MediaLearningPreparationTarget,
        track: &SubtitleTrack,
    ) -> Result<Option<SubtitleTextTrackSnapshot>, ApplicationError> {
        if track.media_id != target.media_id
            || track.status != SubtitleTrackStatus::Available
            || track.language.as_ref().is_none_or(|language| {
                target
                    .requested_learning_language
                    .as_ref()
                    .is_some_and(|requested| requested != language)
            })
            || !track_has_text(track)
        {
            return Ok(None);
        }
        let Some(language) = track.language.clone() else {
            return Ok(None);
        };
        Ok(Some(SubtitleTextTrackSnapshot {
            media_id: track.media_id.clone(),
            track_id: track.id.clone(),
            track_fingerprint: track.fingerprint.clone(),
            text_snapshot_fingerprint: foundation_text_snapshot_fingerprint(track)?,
            language,
        }))
    }
}

impl MediaLearningPreparationInspector for LocalMediaLearningPreparationInspector {
    fn inspect(
        &self,
        target: &MediaLearningPreparationTarget,
        request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, ApplicationError> {
        let analysis = self.services.media_analysis();
        let Some(media) = analysis.read_media(&target.media_id)? else {
            return Err(ApplicationError::NotFound("media"));
        };
        if media.fingerprint != target.media_fingerprint {
            return Err(ApplicationError::Conflict(
                "media learning preparation media snapshot changed",
            ));
        }
        if media.availability != MediaAvailability::Available {
            return Ok(MediaLearningPreparationSourceInspection::Unavailable {
                reason: "media_unavailable".into(),
            });
        }

        if let Some(explicit_id) = request.explicit_subtitle_track_id.as_ref() {
            let Some(track) = analysis.read_subtitle_track(explicit_id)? else {
                return Ok(MediaLearningPreparationSourceInspection::Unavailable {
                    reason: "subtitle_track_not_found".into(),
                });
            };
            return Ok(match self.snapshot(target, &track)? {
                Some(snapshot) => MediaLearningPreparationSourceInspection::Existing { snapshot },
                None => MediaLearningPreparationSourceInspection::Unavailable {
                    reason: "subtitle_track_unavailable".into(),
                },
            });
        }

        let snapshots = analysis
            .list_subtitle_tracks(&target.media_id)?
            .iter()
            .map(|track| self.snapshot(target, track))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        match snapshots.as_slice() {
            [snapshot] => Ok(MediaLearningPreparationSourceInspection::Existing {
                snapshot: snapshot.clone(),
            }),
            [] => Ok(MediaLearningPreparationSourceInspection::Asr {
                audio_track: request.explicit_audio_track,
            }),
            _ => Ok(
                MediaLearningPreparationSourceInspection::SelectionRequired {
                    reason: MediaLearningPreparationSelectionRequired::SubtitleTrack,
                },
            ),
        }
    }
}

fn track_has_text(track: &SubtitleTrack) -> bool {
    track.sentences.iter().any(|sentence| {
        !sentence.original_text.trim().is_empty() || !sentence.display_text.trim().is_empty()
    })
}

#[derive(Debug, Clone)]
enum FoundationChildState {
    Active,
    Completed,
    Failed,
    Cancelled,
}

trait FoundationRuntime: Send + Sync {
    fn prepare(
        &self,
        target: FoundationPreparationTarget,
    ) -> Result<FoundationPreparationChildRef, ApplicationError>;
    fn state(
        &self,
        id: &LearningPreparationRunId,
    ) -> Result<FoundationChildState, ApplicationError>;
    fn cancel(&self, id: &LearningPreparationRunId) -> Result<(), ApplicationError>;
}

impl FoundationRuntime for Arc<LearningPreparationCoordinator> {
    fn prepare(
        &self,
        target: FoundationPreparationTarget,
    ) -> Result<FoundationPreparationChildRef, ApplicationError> {
        let result = LearningPreparationCoordinator::prepare(
            self,
            target,
            FoundationPreparationRequest {
                intent: FoundationPreparationIntent::RecommendedFoundation,
            },
        )?;
        let run = match result {
            PrepareFoundationResult::Run(run) | PrepareFoundationResult::Replaced { run, .. } => {
                run
            }
            PrepareFoundationResult::SelectionRequired(_) => {
                return Err(ApplicationError::Conflict(
                    "foundation preparation unexpectedly requires selection",
                ));
            }
            PrepareFoundationResult::Unavailable(_) => {
                return Err(ApplicationError::Conflict(
                    "foundation preparation is unavailable",
                ));
            }
        };
        Ok(FoundationPreparationChildRef {
            run_id: run.id.clone(),
            input_fingerprint: run.input_fingerprint.clone(),
        })
    }

    fn state(
        &self,
        id: &LearningPreparationRunId,
    ) -> Result<FoundationChildState, ApplicationError> {
        Ok(
            match LearningPreparationCoordinator::get(self, id)?.status {
                LearningPreparationRunStatus::Queued
                | LearningPreparationRunStatus::Running
                | LearningPreparationRunStatus::Cancelling => FoundationChildState::Active,
                LearningPreparationRunStatus::Completed => FoundationChildState::Completed,
                LearningPreparationRunStatus::Failed => FoundationChildState::Failed,
                LearningPreparationRunStatus::Cancelled => FoundationChildState::Cancelled,
            },
        )
    }

    fn cancel(&self, id: &LearningPreparationRunId) -> Result<(), ApplicationError> {
        LearningPreparationCoordinator::cancel(self, id).map(|_| ())
    }
}

#[async_trait]
trait PreparationTranscriptionRuntime: Send + Sync {
    async fn resolve_audio(
        &self,
        target: &MediaLearningPreparationTarget,
        requested_audio_track: Option<MediaAudioTrackIndex>,
    ) -> Result<PreparationAudioSelection, ApplicationError>;
    async fn ensure(
        &self,
        request: EnsurePreparationTranscriptionRequest,
    ) -> Result<TranscriptionJob, ApplicationError>;
    fn job(&self, id: &TranscriptionJobId) -> Result<Option<TranscriptionJob>, ApplicationError>;
    fn cancel(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError>;
}

#[async_trait]
impl PreparationTranscriptionRuntime for Arc<TranscriptionCoordinator> {
    async fn resolve_audio(
        &self,
        target: &MediaLearningPreparationTarget,
        requested_audio_track: Option<MediaAudioTrackIndex>,
    ) -> Result<PreparationAudioSelection, ApplicationError> {
        self.resolve_preparation_audio_track(
            &target.media_id,
            requested_audio_track.map(MediaAudioTrackIndex::as_u32),
        )
        .await
    }

    async fn ensure(
        &self,
        request: EnsurePreparationTranscriptionRequest,
    ) -> Result<TranscriptionJob, ApplicationError> {
        Ok(self.ensure_preparation_transcription(request).await?.job)
    }

    fn job(&self, id: &TranscriptionJobId) -> Result<Option<TranscriptionJob>, ApplicationError> {
        TranscriptionCoordinator::job(self, id)
    }

    fn cancel(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError> {
        self.cancel_job(id).map(|_| ())
    }
}

/// Durable internal parent for the content-level "prepare and start learning"
/// action. It exposes no resource-management or model-selection surface.
pub struct MediaLearningPreparationCoordinator {
    services: AppServices,
    use_cases: Arc<MediaLearningPreparationUseCases>,
    inspector: Arc<dyn MediaLearningPreparationInspector>,
    transcription: Arc<dyn PreparationTranscriptionRuntime>,
    foundation: Arc<dyn FoundationRuntime>,
    active_tasks: Mutex<HashSet<MediaLearningPreparationId>>,
    foundation_start_gate: Mutex<()>,
}

impl MediaLearningPreparationCoordinator {
    pub fn new(
        services: AppServices,
        preparations: Arc<dyn MediaLearningPreparationRepository>,
        transcription: Arc<TranscriptionCoordinator>,
        foundation: Arc<LearningPreparationCoordinator>,
    ) -> Result<Arc<Self>, ApplicationError> {
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        Self::new_with_adapters(
            services,
            preparations,
            inspector,
            Arc::new(transcription),
            Arc::new(foundation),
        )
    }

    fn new_with_adapters(
        services: AppServices,
        preparations: Arc<dyn MediaLearningPreparationRepository>,
        inspector: Arc<dyn MediaLearningPreparationInspector>,
        transcription: Arc<dyn PreparationTranscriptionRuntime>,
        foundation: Arc<dyn FoundationRuntime>,
    ) -> Result<Arc<Self>, ApplicationError> {
        let use_cases = Arc::new(MediaLearningPreparationUseCases::new(
            preparations,
            inspector.clone(),
        ));
        let recovered = use_cases.recover_startup(now_ms())?;
        let coordinator = Arc::new(Self {
            services,
            use_cases,
            inspector,
            transcription,
            foundation,
            active_tasks: Mutex::new(HashSet::new()),
            foundation_start_gate: Mutex::new(()),
        });
        if tokio::runtime::Handle::try_current().is_ok() {
            for run in recovered {
                coordinator.clone().start(run.id);
            }
        }
        Ok(coordinator)
    }

    pub async fn prepare(
        self: &Arc<Self>,
        target: MediaLearningPreparationTarget,
        request: MediaLearningPreparationRequest,
    ) -> Result<PrepareMediaLearningResult, ApplicationError> {
        let inspection = self.inspect_and_resolve_audio(&target, &request).await?;
        let source = {
            let _gate = self
                .foundation_start_gate
                .lock()
                .expect("media preparation foundation gate poisoned");
            let source = self
                .use_cases
                .prepare_resolved(target, request, inspection, now_ms())?;
            if let PrepareMediaLearningResult::Replaced { invalidated, .. } = &source {
                self.cancel_children(invalidated);
            }
            source
        };
        if let PrepareMediaLearningResult::Run(run)
        | PrepareMediaLearningResult::Replaced { run, .. } = &source
        {
            self.clone().start(run.id.clone());
        }
        Ok(source)
    }

    pub fn get(
        &self,
        id: &MediaLearningPreparationId,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        self.use_cases.get(id)
    }

    pub fn cancel(
        self: &Arc<Self>,
        id: &MediaLearningPreparationId,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        let _gate = self
            .foundation_start_gate
            .lock()
            .expect("media preparation foundation gate poisoned");
        let run =
            self.use_cases
                .command(id, MediaLearningPreparationCommand::RequestCancel, now_ms())?;
        self.cancel_children(&run);
        self.clone().start(run.id.clone());
        Ok(run)
    }

    pub fn retry(
        self: &Arc<Self>,
        id: &MediaLearningPreparationId,
    ) -> Result<MediaLearningPreparation, ApplicationError> {
        self.validate_retry_snapshot(&self.use_cases.get(id)?)?;
        let run = self.use_cases.retry(id, now_ms())?;
        self.clone().start(run.id.clone());
        Ok(run)
    }

    async fn inspect_and_resolve_audio(
        &self,
        target: &MediaLearningPreparationTarget,
        request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, ApplicationError> {
        match self.inspector.inspect(target, request)? {
            MediaLearningPreparationSourceInspection::Asr { .. } => {
                match self
                    .transcription
                    .resolve_audio(target, request.explicit_audio_track)
                    .await?
                {
                    PreparationAudioSelection::Selected { audio_track } => {
                        Ok(MediaLearningPreparationSourceInspection::Asr {
                            audio_track: Some(MediaAudioTrackIndex::new(audio_track)),
                        })
                    }
                    PreparationAudioSelection::SelectionRequired { .. } => Ok(
                        MediaLearningPreparationSourceInspection::SelectionRequired {
                            reason: MediaLearningPreparationSelectionRequired::AudioTrack,
                        },
                    ),
                    PreparationAudioSelection::Unavailable { reason } => {
                        Ok(MediaLearningPreparationSourceInspection::Unavailable {
                            reason: reason.into(),
                        })
                    }
                }
            }
            source => Ok(source),
        }
    }

    fn validate_retry_snapshot(
        &self,
        run: &MediaLearningPreparation,
    ) -> Result<(), ApplicationError> {
        let SubtitleTextTrackSlot::Ready { snapshot, .. } = &run.subtitle_text_track else {
            return Ok(());
        };
        let analysis = self.services.media_analysis();
        let media =
            analysis
                .read_media(&run.target.media_id)?
                .ok_or(ApplicationError::Conflict(
                    "media preparation snapshot changed; prepare again",
                ))?;
        let track =
            analysis
                .read_subtitle_track(&snapshot.track_id)?
                .ok_or(ApplicationError::Conflict(
                    "media preparation snapshot changed; prepare again",
                ))?;
        let current_text_fingerprint = foundation_text_snapshot_fingerprint(&track)?;
        if media.fingerprint != run.target.media_fingerprint
            || media.availability != MediaAvailability::Available
            || track.media_id != run.target.media_id
            || track.status != SubtitleTrackStatus::Available
            || track.language.as_ref() != Some(&snapshot.language)
            || run
                .target
                .requested_learning_language
                .as_ref()
                .is_some_and(|language| language != &snapshot.language)
            || track.fingerprint != snapshot.track_fingerprint
            || current_text_fingerprint != snapshot.text_snapshot_fingerprint
        {
            return Err(ApplicationError::Conflict(
                "media preparation snapshot changed; prepare again",
            ));
        }
        Ok(())
    }

    fn start(self: Arc<Self>, id: MediaLearningPreparationId) {
        {
            let mut active = self
                .active_tasks
                .lock()
                .expect("media preparation task mutex poisoned");
            if !active.insert(id.clone()) {
                return;
            }
        }
        tokio::spawn(async move {
            let execution = self.execute(id.clone()).await;
            if let Err(error) = execution {
                let _ = self.record_execution_failure(&id, error.to_string());
            }
            self.active_tasks
                .lock()
                .expect("media preparation task mutex poisoned")
                .remove(&id);
            // A concurrent cancel can be deduplicated while this worker is
            // still registered. Re-check after cleanup so a durable active
            // run is never left without an owner.
            if self
                .use_cases
                .get(&id)
                .is_ok_and(|run| run.status.is_active())
            {
                self.clone().start(id);
            }
        });
    }

    async fn execute(&self, id: MediaLearningPreparationId) -> Result<(), ApplicationError> {
        let mut run = self.use_cases.get(&id)?;
        if run.status == MediaLearningPreparationStatus::Queued {
            run = self
                .use_cases
                .command(&id, MediaLearningPreparationCommand::Start, now_ms())?;
        }
        loop {
            if run.status == MediaLearningPreparationStatus::Cancelling {
                self.cancel_children(&run);
                self.use_cases.command(
                    &id,
                    MediaLearningPreparationCommand::FinishCancellation,
                    now_ms(),
                )?;
                return Ok(());
            }
            if run.status != MediaLearningPreparationStatus::Running {
                return Ok(());
            }

            match run.subtitle_text_track.clone() {
                SubtitleTextTrackSlot::Existing { .. } => {
                    run = self.use_cases.command(
                        &id,
                        MediaLearningPreparationCommand::AcceptExistingSubtitle,
                        now_ms(),
                    )?;
                    continue;
                }
                SubtitleTextTrackSlot::AsrChild {
                    audio_track,
                    job_id: None,
                    ..
                } => {
                    let request = preparation_asr_ensure_request(&run, audio_track)?;
                    let child_id = request.child_id.clone();
                    let job = self.transcription.ensure(request).await?;
                    if job.id != child_id {
                        return Err(ApplicationError::Conflict(
                            "ASR ensure returned another deterministic child",
                        ));
                    }
                    let _gate = self
                        .foundation_start_gate
                        .lock()
                        .expect("media preparation foundation gate poisoned");
                    let current = self.use_cases.get(&id)?;
                    if current.status == MediaLearningPreparationStatus::Cancelling {
                        let _ = self.transcription.cancel(&job.id);
                        self.use_cases.command(
                            &id,
                            MediaLearningPreparationCommand::FinishCancellation,
                            now_ms(),
                        )?;
                        return Ok(());
                    }
                    if current.status != MediaLearningPreparationStatus::Running {
                        let _ = self.transcription.cancel(&job.id);
                        return Ok(());
                    }
                    run = self.use_cases.command(
                        &id,
                        MediaLearningPreparationCommand::AttachAsrChild {
                            job_id: job.id,
                            input_provenance_fingerprint: job.input_fingerprint,
                        },
                        now_ms(),
                    )?;
                    continue;
                }
                SubtitleTextTrackSlot::AsrChild {
                    audio_track,
                    job_id: Some(job_id),
                    input_provenance_fingerprint,
                    ..
                } => {
                    let job = self
                        .transcription
                        .job(&job_id)?
                        .ok_or(ApplicationError::NotFound("transcription job"))?;
                    if input_provenance_fingerprint.as_deref()
                        != Some(job.input_fingerprint.as_str())
                    {
                        return Err(ApplicationError::Conflict(
                            "ASR child input provenance changed",
                        ));
                    }
                    match job.status {
                        TranscriptionJobStatus::Queued
                        | TranscriptionJobStatus::Extracting
                        | TranscriptionJobStatus::Transcribing
                        | TranscriptionJobStatus::Importing => {
                            tokio::time::sleep(POLL_INTERVAL).await;
                            run = self.use_cases.get(&id)?;
                        }
                        TranscriptionJobStatus::Completed => {
                            let snapshot = self.freeze_asr_snapshot(&run, &job)?;
                            run = self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::CompleteAsrChild {
                                    job_id,
                                    snapshot,
                                },
                                now_ms(),
                            )?;
                        }
                        TranscriptionJobStatus::Failed
                            if job.error_code.as_deref() == Some("interrupted") =>
                        {
                            let request = preparation_asr_ensure_request(&run, audio_track)?;
                            let child_id = request.child_id.clone();
                            if child_id != job_id {
                                return Err(ApplicationError::Conflict(
                                    "attached ASR child identity changed",
                                ));
                            }
                            let restarted = self.transcription.ensure(request).await?;
                            if restarted.id != job_id
                                || input_provenance_fingerprint.as_deref()
                                    != Some(restarted.input_fingerprint.as_str())
                            {
                                return Err(ApplicationError::Conflict(
                                    "restarted ASR child input provenance changed",
                                ));
                            }
                            let _gate = self
                                .foundation_start_gate
                                .lock()
                                .expect("media preparation foundation gate poisoned");
                            let current = self.use_cases.get(&id)?;
                            if current.status == MediaLearningPreparationStatus::Cancelling {
                                let _ = self.transcription.cancel(&restarted.id);
                                self.use_cases.command(
                                    &id,
                                    MediaLearningPreparationCommand::FinishCancellation,
                                    now_ms(),
                                )?;
                                return Ok(());
                            }
                            if current.status != MediaLearningPreparationStatus::Running {
                                let _ = self.transcription.cancel(&restarted.id);
                                return Ok(());
                            }
                            run = current;
                        }
                        TranscriptionJobStatus::Failed => {
                            self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::FailAsrChild {
                                    job_id,
                                    reason: job
                                        .error_code
                                        .map(|code| format!("asr_failed:{code}"))
                                        .unwrap_or_else(|| "asr_failed".into()),
                                },
                                now_ms(),
                            )?;
                            return Ok(());
                        }
                        TranscriptionJobStatus::Cancelled => {
                            self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::FailAsrChild {
                                    job_id,
                                    reason: "asr_cancelled".into(),
                                },
                                now_ms(),
                            )?;
                            return Ok(());
                        }
                    }
                    continue;
                }
                SubtitleTextTrackSlot::Ready { .. } => {}
                SubtitleTextTrackSlot::Failed { .. } | SubtitleTextTrackSlot::Cancelled => {
                    return Ok(());
                }
            }

            match run.foundation.clone() {
                FoundationPreparationSlot::Pending => {
                    let _gate = self
                        .foundation_start_gate
                        .lock()
                        .expect("media preparation foundation gate poisoned");
                    run = self.use_cases.get(&id)?;
                    if run.status != MediaLearningPreparationStatus::Running {
                        continue;
                    }
                    let target = run.foundation_target().ok_or(ApplicationError::Conflict(
                        "foundation preparation requires a ready subtitle snapshot",
                    ))?;
                    let child = self.foundation.prepare(target)?;
                    run = self.use_cases.command(
                        &id,
                        MediaLearningPreparationCommand::AttachFoundationChild { child },
                        now_ms(),
                    )?;
                }
                FoundationPreparationSlot::Child { child } => {
                    match self.foundation.state(&child.run_id)? {
                        FoundationChildState::Active => {
                            tokio::time::sleep(POLL_INTERVAL).await;
                            run = self.use_cases.get(&id)?;
                        }
                        FoundationChildState::Completed => {
                            self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::CompleteFoundationChild {
                                    run_id: child.run_id,
                                },
                                now_ms(),
                            )?;
                            return Ok(());
                        }
                        FoundationChildState::Failed => {
                            self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::FailFoundationChild {
                                    run_id: child.run_id,
                                    reason: "foundation_failed".into(),
                                },
                                now_ms(),
                            )?;
                            return Ok(());
                        }
                        FoundationChildState::Cancelled => {
                            self.use_cases.command(
                                &id,
                                MediaLearningPreparationCommand::FailFoundationChild {
                                    run_id: child.run_id,
                                    reason: "foundation_cancelled".into(),
                                },
                                now_ms(),
                            )?;
                            return Ok(());
                        }
                    }
                }
                FoundationPreparationSlot::Ready { .. }
                | FoundationPreparationSlot::Failed { .. }
                | FoundationPreparationSlot::Cancelled => return Ok(()),
            }
        }
    }

    fn freeze_asr_snapshot(
        &self,
        run: &MediaLearningPreparation,
        job: &TranscriptionJob,
    ) -> Result<SubtitleTextTrackSnapshot, ApplicationError> {
        if job.media_id != run.target.media_id
            || job.media_fingerprint != run.target.media_fingerprint
        {
            return Err(ApplicationError::Conflict(
                "ASR child media snapshot changed",
            ));
        }
        let track_id = job.generated_track_id.as_ref().ok_or_else(|| {
            ApplicationError::Repository("ASR completed without a subtitle track".into())
        })?;
        let track = self
            .services
            .media_analysis()
            .read_subtitle_track(track_id)?
            .ok_or(ApplicationError::NotFound("ASR subtitle track"))?;
        let inspector = LocalMediaLearningPreparationInspector {
            services: self.services.clone(),
        };
        inspector.snapshot(&run.target, &track)?.ok_or({
            ApplicationError::Conflict("ASR subtitle track is not a valid text snapshot")
        })
    }

    fn cancel_children(&self, run: &MediaLearningPreparation) {
        if let SubtitleTextTrackSlot::AsrChild { job_id, .. } = &run.subtitle_text_track {
            let child_id = match job_id.clone() {
                Some(job_id) => Some(job_id),
                None => preparation_asr_child_identity(run)
                    .ok()
                    .map(|identity| identity.child_id),
            };
            if let Some(child_id) = child_id {
                let should_cancel = match self.transcription.job(&child_id) {
                    Ok(Some(job)) => matches!(
                        job.status,
                        TranscriptionJobStatus::Queued
                            | TranscriptionJobStatus::Extracting
                            | TranscriptionJobStatus::Transcribing
                    ),
                    Ok(None) => false,
                    Err(_) => true,
                };
                if should_cancel {
                    let _ = self.transcription.cancel(&child_id);
                }
            }
        }
        if let FoundationPreparationSlot::Child { child } = &run.foundation {
            let _ = self.foundation.cancel(&child.run_id);
        }
    }

    fn record_execution_failure(
        &self,
        id: &MediaLearningPreparationId,
        reason: String,
    ) -> Result<(), ApplicationError> {
        let run = self.use_cases.get(id)?;
        if !matches!(
            run.status,
            MediaLearningPreparationStatus::Queued | MediaLearningPreparationStatus::Running
        ) {
            return Ok(());
        }
        self.use_cases.command(
            id,
            MediaLearningPreparationCommand::FailExecution { reason },
            now_ms(),
        )?;
        Ok(())
    }
}

struct PreparationAsrChildIdentity {
    idempotency_key: String,
    child_id: TranscriptionJobId,
}

fn preparation_asr_child_identity(
    run: &MediaLearningPreparation,
) -> Result<PreparationAsrChildIdentity, ApplicationError> {
    let idempotency_key = format!(
        "media-learning-preparation-input:{}:asr",
        run.input_fingerprint
    );
    let child_id = preparation_transcription_child_id(&idempotency_key)?;
    Ok(PreparationAsrChildIdentity {
        idempotency_key,
        child_id,
    })
}

fn preparation_asr_ensure_request(
    run: &MediaLearningPreparation,
    audio_track: MediaAudioTrackIndex,
) -> Result<EnsurePreparationTranscriptionRequest, ApplicationError> {
    let identity = preparation_asr_child_identity(run)?;
    Ok(EnsurePreparationTranscriptionRequest {
        idempotency_key: identity.idempotency_key,
        child_id: identity.child_id,
        media_id: run.target.media_id.clone(),
        language: run
            .target
            .requested_learning_language
            .as_ref()
            .map(|language| language.as_str().to_owned()),
        audio_track: audio_track.as_u32(),
        terminal_policy: if run.retry_of_id.is_some() {
            PreparationTranscriptionTerminalPolicy::Restart
        } else {
            PreparationTranscriptionTerminalPolicy::Preserve
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use application::{ImportSubtitle, RegisterMedia};
    use domain::{
        LanguageCode, MediaId, MediaKind, SubtitleTrack, TranscriptionDestination,
        TranscriptionModelId, TranscriptionPurpose,
    };
    use persistence_sqlite::SqliteRepository;
    use tokio::sync::Notify;

    use super::*;

    struct ScriptedTranscription {
        audio: PreparationAudioSelection,
        resolve_calls: AtomicUsize,
        ensure_calls: AtomicUsize,
        idempotency_keys: Mutex<Vec<String>>,
        terminal_policies: Mutex<Vec<PreparationTranscriptionTerminalPolicy>>,
    }

    #[async_trait]
    impl PreparationTranscriptionRuntime for ScriptedTranscription {
        async fn resolve_audio(
            &self,
            _target: &MediaLearningPreparationTarget,
            _requested_audio_track: Option<MediaAudioTrackIndex>,
        ) -> Result<PreparationAudioSelection, ApplicationError> {
            self.resolve_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.audio.clone())
        }

        async fn ensure(
            &self,
            request: EnsurePreparationTranscriptionRequest,
        ) -> Result<TranscriptionJob, ApplicationError> {
            self.ensure_calls.fetch_add(1, Ordering::Relaxed);
            self.idempotency_keys
                .lock()
                .unwrap()
                .push(request.idempotency_key.clone());
            self.terminal_policies
                .lock()
                .unwrap()
                .push(request.terminal_policy);
            Err(ApplicationError::Repository(
                "scripted ASR must not start".into(),
            ))
        }

        fn job(
            &self,
            _id: &TranscriptionJobId,
        ) -> Result<Option<TranscriptionJob>, ApplicationError> {
            Ok(None)
        }

        fn cancel(&self, _id: &TranscriptionJobId) -> Result<(), ApplicationError> {
            Ok(())
        }
    }

    struct RecoveringTranscription {
        job: Mutex<TranscriptionJob>,
        ensure_requests: Mutex<Vec<EnsurePreparationTranscriptionRequest>>,
        ensure_called: Notify,
        cancel_calls: AtomicUsize,
    }

    #[async_trait]
    impl PreparationTranscriptionRuntime for RecoveringTranscription {
        async fn resolve_audio(
            &self,
            _target: &MediaLearningPreparationTarget,
            _requested_audio_track: Option<MediaAudioTrackIndex>,
        ) -> Result<PreparationAudioSelection, ApplicationError> {
            unreachable!("startup recovery already has a resolved audio track")
        }

        async fn ensure(
            &self,
            request: EnsurePreparationTranscriptionRequest,
        ) -> Result<TranscriptionJob, ApplicationError> {
            self.ensure_requests.lock().unwrap().push(request.clone());
            let mut job = self.job.lock().unwrap();
            assert_eq!(request.child_id, job.id);
            assert_eq!(request.media_id, job.media_id);
            assert_eq!(request.audio_track, job.audio_track.unwrap());
            job.status = TranscriptionJobStatus::Queued;
            job.error_code = None;
            job.error_message = None;
            let restarted = job.clone();
            drop(job);
            self.ensure_called.notify_one();
            Ok(restarted)
        }

        fn job(
            &self,
            id: &TranscriptionJobId,
        ) -> Result<Option<TranscriptionJob>, ApplicationError> {
            let job = self.job.lock().unwrap().clone();
            Ok((job.id == *id).then_some(job))
        }

        fn cancel(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError> {
            let mut job = self.job.lock().unwrap();
            assert_eq!(id, &job.id);
            job.status = TranscriptionJobStatus::Cancelled;
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct BlockingTranscription {
        media_fingerprint: String,
        ensure_entered: Notify,
        ensure_release: Notify,
        cancel_calls: AtomicUsize,
    }

    struct ActiveTranscription {
        media_fingerprint: String,
        jobs: Mutex<HashMap<TranscriptionJobId, TranscriptionJob>>,
        cancelled: Mutex<Vec<TranscriptionJobId>>,
    }

    struct CrashWindowTranscription {
        job: Mutex<TranscriptionJob>,
        ensure_calls: AtomicUsize,
        cancel_calls: AtomicUsize,
    }

    #[async_trait]
    impl PreparationTranscriptionRuntime for CrashWindowTranscription {
        async fn resolve_audio(
            &self,
            _target: &MediaLearningPreparationTarget,
            _requested_audio_track: Option<MediaAudioTrackIndex>,
        ) -> Result<PreparationAudioSelection, ApplicationError> {
            unreachable!("the durable parent already has a resolved audio track")
        }

        async fn ensure(
            &self,
            request: EnsurePreparationTranscriptionRequest,
        ) -> Result<TranscriptionJob, ApplicationError> {
            self.ensure_calls.fetch_add(1, Ordering::Relaxed);
            let job = self.job.lock().unwrap().clone();
            assert_eq!(request.child_id, job.id);
            Ok(job)
        }

        fn job(
            &self,
            id: &TranscriptionJobId,
        ) -> Result<Option<TranscriptionJob>, ApplicationError> {
            let job = self.job.lock().unwrap().clone();
            Ok((job.id == *id).then_some(job))
        }

        fn cancel(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError> {
            let mut job = self.job.lock().unwrap();
            if job.id == *id
                && matches!(
                    job.status,
                    TranscriptionJobStatus::Queued
                        | TranscriptionJobStatus::Extracting
                        | TranscriptionJobStatus::Transcribing
                )
            {
                job.status = TranscriptionJobStatus::Cancelled;
                self.cancel_calls.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        }
    }

    #[async_trait]
    impl PreparationTranscriptionRuntime for ActiveTranscription {
        async fn resolve_audio(
            &self,
            _target: &MediaLearningPreparationTarget,
            _requested_audio_track: Option<MediaAudioTrackIndex>,
        ) -> Result<PreparationAudioSelection, ApplicationError> {
            Ok(PreparationAudioSelection::Selected { audio_track: 0 })
        }

        async fn ensure(
            &self,
            request: EnsurePreparationTranscriptionRequest,
        ) -> Result<TranscriptionJob, ApplicationError> {
            let job = transcription_job(
                request.child_id,
                request.media_id,
                &self.media_fingerprint,
                request.language,
                request.audio_track,
                &format!("active-input:{}", request.idempotency_key),
                TranscriptionJobStatus::Queued,
            );
            self.jobs
                .lock()
                .unwrap()
                .insert(job.id.clone(), job.clone());
            Ok(job)
        }

        fn job(
            &self,
            id: &TranscriptionJobId,
        ) -> Result<Option<TranscriptionJob>, ApplicationError> {
            Ok(self.jobs.lock().unwrap().get(id).cloned())
        }

        fn cancel(&self, id: &TranscriptionJobId) -> Result<(), ApplicationError> {
            if let Some(job) = self.jobs.lock().unwrap().get_mut(id) {
                job.status = TranscriptionJobStatus::Cancelled;
            }
            self.cancelled.lock().unwrap().push(id.clone());
            Ok(())
        }
    }

    #[async_trait]
    impl PreparationTranscriptionRuntime for BlockingTranscription {
        async fn resolve_audio(
            &self,
            _target: &MediaLearningPreparationTarget,
            _requested_audio_track: Option<MediaAudioTrackIndex>,
        ) -> Result<PreparationAudioSelection, ApplicationError> {
            Ok(PreparationAudioSelection::Selected { audio_track: 0 })
        }

        async fn ensure(
            &self,
            request: EnsurePreparationTranscriptionRequest,
        ) -> Result<TranscriptionJob, ApplicationError> {
            let job = transcription_job(
                request.child_id,
                request.media_id,
                &self.media_fingerprint,
                request.language,
                request.audio_track,
                "blocking-input-provenance",
                TranscriptionJobStatus::Queued,
            );
            self.ensure_entered.notify_one();
            self.ensure_release.notified().await;
            Ok(job)
        }

        fn job(
            &self,
            _id: &TranscriptionJobId,
        ) -> Result<Option<TranscriptionJob>, ApplicationError> {
            Ok(None)
        }

        fn cancel(&self, _id: &TranscriptionJobId) -> Result<(), ApplicationError> {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct ScriptedFoundation {
        prepare_calls: AtomicUsize,
    }

    impl FoundationRuntime for ScriptedFoundation {
        fn prepare(
            &self,
            target: FoundationPreparationTarget,
        ) -> Result<FoundationPreparationChildRef, ApplicationError> {
            self.prepare_calls.fetch_add(1, Ordering::Relaxed);
            Ok(FoundationPreparationChildRef {
                run_id: LearningPreparationRunId::from_fingerprint(
                    target.subtitle_text_fingerprint.as_str(),
                ),
                input_fingerprint: target.input_fingerprint(),
            })
        }

        fn state(
            &self,
            _id: &LearningPreparationRunId,
        ) -> Result<FoundationChildState, ApplicationError> {
            Ok(FoundationChildState::Completed)
        }

        fn cancel(&self, _id: &LearningPreparationRunId) -> Result<(), ApplicationError> {
            Ok(())
        }
    }

    fn setup() -> (Arc<SqliteRepository>, AppServices, domain::MediaItem) {
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
        let media = services
            .media_analysis()
            .register_media(RegisterMedia {
                path: "/test/media.mkv".into(),
                fingerprint: "media-fingerprint".into(),
                title: "Test".into(),
                kind: MediaKind::Video,
                duration_ms: Some(1_000),
            })
            .unwrap();
        (repository, services, media)
    }

    fn import_track(
        services: &AppServices,
        media_id: &MediaId,
        source: &str,
        identity_salt: Option<&str>,
        language: &str,
    ) -> SubtitleTrack {
        services
            .media_analysis()
            .import_subtitle(ImportSubtitle {
                media_id: media_id.clone(),
                source_name: source.into(),
                content: b"1\n00:00:00,000 --> 00:00:01,000\nHello world\n".to_vec(),
                language: Some(language.into()),
                identity_salt: identity_salt.map(str::to_owned),
            })
            .unwrap()
    }

    fn target(media: &domain::MediaItem) -> MediaLearningPreparationTarget {
        MediaLearningPreparationTarget {
            media_id: media.id.clone(),
            media_fingerprint: media.fingerprint.clone(),
            requested_learning_language: Some(LanguageCode::parse("en").unwrap()),
        }
    }

    fn transcription_job(
        id: TranscriptionJobId,
        media_id: MediaId,
        media_fingerprint: &str,
        requested_language: Option<String>,
        audio_track: u32,
        input_fingerprint: &str,
        status: TranscriptionJobStatus,
    ) -> TranscriptionJob {
        TranscriptionJob {
            id,
            media_id,
            media_title: "Test".into(),
            media_fingerprint: media_fingerprint.into(),
            provider_id: "whisper.cpp".into(),
            provider_version: "test".into(),
            runtime_id: "whisper.cpp".into(),
            runtime_version: "test".into(),
            model_id: TranscriptionModelId::parse("whisper.cpp:base@main").unwrap(),
            model_revision: "main".into(),
            model_checksum_sha256: "model-checksum".into(),
            destination: TranscriptionDestination::Primary,
            purpose: TranscriptionPurpose::Transcribe,
            requested_language,
            detected_language: None,
            audio_track: Some(audio_track),
            settings_json: "{}".into(),
            input_fingerprint: input_fingerprint.into(),
            status,
            phase_progress: 0,
            error_code: None,
            error_message: None,
            retry_of_job_id: None,
            generated_track_id: None,
            created_at_ms: 1,
            started_at_ms: None,
            completed_at_ms: None,
            updated_at_ms: 1,
            archived_at_ms: None,
        }
    }

    fn request() -> MediaLearningPreparationRequest {
        MediaLearningPreparationRequest {
            explicit_subtitle_track_id: None,
            explicit_audio_track: None,
        }
    }

    fn coordinator(
        repository: Arc<SqliteRepository>,
        services: AppServices,
        transcription: Arc<ScriptedTranscription>,
        foundation: Arc<ScriptedFoundation>,
    ) -> Arc<MediaLearningPreparationCoordinator> {
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription,
            foundation,
        )
        .unwrap()
    }

    async fn wait_terminal(
        coordinator: &MediaLearningPreparationCoordinator,
        id: &MediaLearningPreparationId,
    ) -> MediaLearningPreparation {
        for _ in 0..100 {
            let run = coordinator.get(id).unwrap();
            if !run.status.is_active() {
                return run;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("media preparation did not become terminal");
    }

    async fn wait_asr_attached(
        coordinator: &MediaLearningPreparationCoordinator,
        id: &MediaLearningPreparationId,
    ) -> TranscriptionJobId {
        for _ in 0..100 {
            if let SubtitleTextTrackSlot::AsrChild {
                job_id: Some(job_id),
                ..
            } = coordinator.get(id).unwrap().subtitle_text_track
            {
                return job_id;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("media preparation did not attach its ASR child");
    }

    #[tokio::test]
    async fn existing_text_track_skips_asr_and_bridges_to_foundation() {
        let (repository, services, media) = setup();
        import_track(&services, &media.id, "existing.srt", None, "en");
        let transcription = Arc::new(ScriptedTranscription {
            audio: PreparationAudioSelection::Selected { audio_track: 0 },
            resolve_calls: AtomicUsize::new(0),
            ensure_calls: AtomicUsize::new(0),
            idempotency_keys: Mutex::new(Vec::new()),
            terminal_policies: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let coordinator = coordinator(
            repository,
            services,
            transcription.clone(),
            foundation.clone(),
        );

        let PrepareMediaLearningResult::Run(created) = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap()
        else {
            panic!("expected durable parent run");
        };
        let completed = wait_terminal(&coordinator, &created.id).await;

        assert_eq!(completed.status, MediaLearningPreparationStatus::Completed);
        assert_eq!(transcription.resolve_calls.load(Ordering::Relaxed), 0);
        assert_eq!(transcription.ensure_calls.load(Ordering::Relaxed), 0);
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn multiple_eligible_subtitles_require_selection_without_children() {
        let (repository, services, media) = setup();
        import_track(&services, &media.id, "one.srt", Some("one"), "en");
        import_track(&services, &media.id, "two.srt", Some("two"), "en");
        let transcription = Arc::new(ScriptedTranscription {
            audio: PreparationAudioSelection::Selected { audio_track: 0 },
            resolve_calls: AtomicUsize::new(0),
            ensure_calls: AtomicUsize::new(0),
            idempotency_keys: Mutex::new(Vec::new()),
            terminal_policies: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let coordinator = coordinator(
            repository,
            services,
            transcription.clone(),
            foundation.clone(),
        );

        let result = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap();

        assert_eq!(
            result,
            PrepareMediaLearningResult::SelectionRequired(
                MediaLearningPreparationSelectionRequired::SubtitleTrack
            )
        );
        assert_eq!(transcription.resolve_calls.load(Ordering::Relaxed), 0);
        assert_eq!(transcription.ensure_calls.load(Ordering::Relaxed), 0);
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn ambiguous_audio_preflight_creates_no_asr_child_or_parent_run() {
        let (repository, services, media) = setup();
        let transcription = Arc::new(ScriptedTranscription {
            audio: PreparationAudioSelection::SelectionRequired {
                reason: "audio_track_required",
            },
            resolve_calls: AtomicUsize::new(0),
            ensure_calls: AtomicUsize::new(0),
            idempotency_keys: Mutex::new(Vec::new()),
            terminal_policies: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let coordinator = coordinator(
            repository,
            services,
            transcription.clone(),
            foundation.clone(),
        );

        let result = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap();

        assert_eq!(
            result,
            PrepareMediaLearningResult::SelectionRequired(
                MediaLearningPreparationSelectionRequired::AudioTrack
            )
        );
        assert_eq!(transcription.resolve_calls.load(Ordering::Relaxed), 1);
        assert_eq!(transcription.ensure_calls.load(Ordering::Relaxed), 0);
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn retry_reuses_the_same_asr_child_identity() {
        let (repository, services, media) = setup();
        let transcription = Arc::new(ScriptedTranscription {
            audio: PreparationAudioSelection::Selected { audio_track: 0 },
            resolve_calls: AtomicUsize::new(0),
            ensure_calls: AtomicUsize::new(0),
            idempotency_keys: Mutex::new(Vec::new()),
            terminal_policies: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let coordinator = coordinator(repository, services, transcription.clone(), foundation);

        let PrepareMediaLearningResult::Run(first) = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap()
        else {
            panic!("expected first durable parent");
        };
        let failed = wait_terminal(&coordinator, &first.id).await;
        assert_eq!(failed.status, MediaLearningPreparationStatus::Failed);

        let retry = coordinator.retry(&first.id).unwrap();
        let retry_failed = wait_terminal(&coordinator, &retry.id).await;
        assert_eq!(retry_failed.status, MediaLearningPreparationStatus::Failed);

        let keys = transcription.idempotency_keys.lock().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], keys[1]);
        assert_eq!(
            preparation_transcription_child_id(&keys[0]).unwrap(),
            preparation_transcription_child_id(&keys[1]).unwrap()
        );
        drop(keys);
        assert_eq!(
            *transcription.terminal_policies.lock().unwrap(),
            vec![
                PreparationTranscriptionTerminalPolicy::Preserve,
                PreparationTranscriptionTerminalPolicy::Restart,
            ]
        );
    }

    #[test]
    fn retry_rejects_an_archived_ready_subtitle_snapshot() {
        let (repository, services, media) = setup();
        let track = import_track(&services, &media.id, "existing.srt", None, "en");
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        let use_cases =
            MediaLearningPreparationUseCases::new(repository.clone(), inspector.clone());
        let PrepareMediaLearningResult::Run(created) =
            use_cases.prepare(target(&media), request(), 10).unwrap()
        else {
            panic!("expected durable parent run");
        };
        use_cases
            .command(&created.id, MediaLearningPreparationCommand::Start, 11)
            .unwrap();
        use_cases
            .command(
                &created.id,
                MediaLearningPreparationCommand::AcceptExistingSubtitle,
                12,
            )
            .unwrap();
        let foundation_child = FoundationPreparationChildRef {
            run_id: LearningPreparationRunId::parse("failed-foundation").unwrap(),
            input_fingerprint: "failed-foundation-input".into(),
        };
        use_cases
            .command(
                &created.id,
                MediaLearningPreparationCommand::AttachFoundationChild {
                    child: foundation_child.clone(),
                },
                13,
            )
            .unwrap();
        use_cases
            .command(
                &created.id,
                MediaLearningPreparationCommand::FailFoundationChild {
                    run_id: foundation_child.run_id,
                    reason: "foundation_failed".into(),
                },
                14,
            )
            .unwrap();
        services
            .media_analysis()
            .archive_subtitle_track(&track.id)
            .unwrap();
        let transcription = Arc::new(ScriptedTranscription {
            audio: PreparationAudioSelection::Selected { audio_track: 0 },
            resolve_calls: AtomicUsize::new(0),
            ensure_calls: AtomicUsize::new(0),
            idempotency_keys: Mutex::new(Vec::new()),
            terminal_policies: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let coordinator = MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription,
            foundation,
        )
        .unwrap();

        assert!(matches!(
            coordinator.retry(&created.id),
            Err(ApplicationError::Conflict(
                "media preparation snapshot changed; prepare again"
            ))
        ));
    }

    #[tokio::test]
    async fn replacing_an_active_parent_cancels_its_attached_asr_child() {
        let (repository, services, media) = setup();
        let transcription = Arc::new(ActiveTranscription {
            media_fingerprint: media.fingerprint.clone(),
            jobs: Mutex::new(HashMap::new()),
            cancelled: Mutex::new(Vec::new()),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        let coordinator = MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription.clone(),
            foundation.clone(),
        )
        .unwrap();
        let PrepareMediaLearningResult::Run(first) = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap()
        else {
            panic!("expected first durable parent");
        };
        let first_job_id = wait_asr_attached(&coordinator, &first.id).await;

        let replacement_request = MediaLearningPreparationRequest {
            explicit_subtitle_track_id: None,
            explicit_audio_track: Some(MediaAudioTrackIndex::new(0)),
        };
        let PrepareMediaLearningResult::Replaced { run, invalidated } = coordinator
            .prepare(target(&media), replacement_request)
            .await
            .unwrap()
        else {
            panic!("expected replacement parent");
        };

        assert_eq!(invalidated.id, first.id);
        assert!(matches!(
            invalidated.subtitle_text_track,
            SubtitleTextTrackSlot::AsrChild {
                job_id: Some(ref job_id),
                ..
            } if job_id == &first_job_id
        ));
        assert!(
            transcription
                .cancelled
                .lock()
                .unwrap()
                .contains(&first_job_id)
        );
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            coordinator.get(&first.id).unwrap().status,
            MediaLearningPreparationStatus::Failed
        );

        let replacement_job_id = wait_asr_attached(&coordinator, &run.id).await;
        assert_ne!(replacement_job_id, first_job_id);
        coordinator.cancel(&run.id).unwrap();
        let cancelled = wait_terminal(&coordinator, &run.id).await;
        assert_eq!(cancelled.status, MediaLearningPreparationStatus::Cancelled);
        assert!(
            transcription
                .cancelled
                .lock()
                .unwrap()
                .contains(&replacement_job_id)
        );
    }

    #[tokio::test]
    async fn startup_reensures_an_attached_interrupted_asr_child_without_changing_provenance() {
        let (repository, services, media) = setup();
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        let use_cases =
            MediaLearningPreparationUseCases::new(repository.clone(), inspector.clone());
        let PrepareMediaLearningResult::Run(created) = use_cases
            .prepare_resolved(
                target(&media),
                request(),
                MediaLearningPreparationSourceInspection::Asr {
                    audio_track: Some(MediaAudioTrackIndex::new(0)),
                },
                10,
            )
            .unwrap()
        else {
            panic!("expected durable parent run");
        };
        let running = use_cases
            .command(&created.id, MediaLearningPreparationCommand::Start, 11)
            .unwrap();
        let identity = preparation_asr_child_identity(&running).unwrap();
        let idempotency_key = identity.idempotency_key;
        let job_id = identity.child_id;
        let input_fingerprint = "stable-asr-input";
        use_cases
            .command(
                &running.id,
                MediaLearningPreparationCommand::AttachAsrChild {
                    job_id: job_id.clone(),
                    input_provenance_fingerprint: input_fingerprint.into(),
                },
                12,
            )
            .unwrap();
        let mut interrupted = transcription_job(
            job_id.clone(),
            media.id.clone(),
            &media.fingerprint,
            Some("en".into()),
            0,
            input_fingerprint,
            TranscriptionJobStatus::Failed,
        );
        interrupted.error_code = Some("interrupted".into());
        interrupted.error_message = Some("service stopped".into());
        let transcription = Arc::new(RecoveringTranscription {
            job: Mutex::new(interrupted),
            ensure_requests: Mutex::new(Vec::new()),
            ensure_called: Notify::new(),
            cancel_calls: AtomicUsize::new(0),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });

        let coordinator = MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription.clone(),
            foundation.clone(),
        )
        .unwrap();
        tokio::time::timeout(
            Duration::from_secs(1),
            transcription.ensure_called.notified(),
        )
        .await
        .expect("startup must re-ensure the interrupted child");

        let recovered = coordinator.get(&created.id).unwrap();
        assert_eq!(recovered.status, MediaLearningPreparationStatus::Running);
        assert!(matches!(
            recovered.subtitle_text_track,
            SubtitleTextTrackSlot::AsrChild {
                audio_track,
                job_id: Some(_),
                input_provenance_fingerprint: Some(ref fingerprint),
            } if audio_track == MediaAudioTrackIndex::new(0)
                && fingerprint == input_fingerprint
        ));
        {
            let requests = transcription.ensure_requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].idempotency_key, idempotency_key);
        }
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);

        coordinator.cancel(&created.id).unwrap();
        let cancelled = wait_terminal(&coordinator, &created.id).await;
        assert_eq!(cancelled.status, MediaLearningPreparationStatus::Cancelled);
    }

    #[tokio::test]
    async fn startup_cancel_derives_and_cancels_an_asr_child_created_before_attach() {
        let (repository, services, media) = setup();
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        let use_cases =
            MediaLearningPreparationUseCases::new(repository.clone(), inspector.clone());
        let PrepareMediaLearningResult::Run(created) = use_cases
            .prepare_resolved(
                target(&media),
                request(),
                MediaLearningPreparationSourceInspection::Asr {
                    audio_track: Some(MediaAudioTrackIndex::new(0)),
                },
                10,
            )
            .unwrap()
        else {
            panic!("expected durable parent run");
        };
        let running = use_cases
            .command(&created.id, MediaLearningPreparationCommand::Start, 11)
            .unwrap();
        assert!(matches!(
            running.subtitle_text_track,
            SubtitleTextTrackSlot::AsrChild { job_id: None, .. }
        ));
        let identity = preparation_asr_child_identity(&running).unwrap();
        let child = transcription_job(
            identity.child_id,
            media.id.clone(),
            &media.fingerprint,
            Some("en".into()),
            0,
            "created-before-parent-attach",
            TranscriptionJobStatus::Queued,
        );
        let transcription = Arc::new(CrashWindowTranscription {
            job: Mutex::new(child),
            ensure_calls: AtomicUsize::new(0),
            cancel_calls: AtomicUsize::new(0),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });

        let coordinator = MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription.clone(),
            foundation.clone(),
        )
        .unwrap();
        let cancelling = coordinator.cancel(&created.id).unwrap();
        assert_eq!(
            cancelling.status,
            MediaLearningPreparationStatus::Cancelling
        );
        let cancelled = wait_terminal(&coordinator, &created.id).await;

        assert_eq!(cancelled.status, MediaLearningPreparationStatus::Cancelled);
        assert_eq!(transcription.ensure_calls.load(Ordering::Relaxed), 0);
        assert_eq!(transcription.cancel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            transcription.job.lock().unwrap().status,
            TranscriptionJobStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn cancellation_while_ensure_is_blocked_cancels_returned_child_and_finishes_parent() {
        let (repository, services, media) = setup();
        let transcription = Arc::new(BlockingTranscription {
            media_fingerprint: media.fingerprint.clone(),
            ensure_entered: Notify::new(),
            ensure_release: Notify::new(),
            cancel_calls: AtomicUsize::new(0),
        });
        let foundation = Arc::new(ScriptedFoundation {
            prepare_calls: AtomicUsize::new(0),
        });
        let inspector = Arc::new(LocalMediaLearningPreparationInspector {
            services: services.clone(),
        });
        let coordinator = MediaLearningPreparationCoordinator::new_with_adapters(
            services,
            repository,
            inspector,
            transcription.clone(),
            foundation.clone(),
        )
        .unwrap();

        let PrepareMediaLearningResult::Run(created) = coordinator
            .prepare(target(&media), request())
            .await
            .unwrap()
        else {
            panic!("expected durable parent run");
        };
        tokio::time::timeout(
            Duration::from_secs(1),
            transcription.ensure_entered.notified(),
        )
        .await
        .expect("worker must enter ASR ensure");

        let cancelling = coordinator.cancel(&created.id).unwrap();
        assert_eq!(
            cancelling.status,
            MediaLearningPreparationStatus::Cancelling
        );
        transcription.ensure_release.notify_one();
        let cancelled = wait_terminal(&coordinator, &created.id).await;

        assert_eq!(cancelled.status, MediaLearningPreparationStatus::Cancelled);
        assert_eq!(transcription.cancel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(foundation.prepare_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn explicit_subtitle_must_match_the_requested_learning_language() {
        let (_repository, services, media) = setup();
        let zh = import_track(&services, &media.id, "zh.srt", None, "zh");
        let inspector = LocalMediaLearningPreparationInspector { services };

        let result = inspector
            .inspect(
                &target(&media),
                &MediaLearningPreparationRequest {
                    explicit_subtitle_track_id: Some(zh.id),
                    explicit_audio_track: None,
                },
            )
            .unwrap();

        assert_eq!(
            result,
            MediaLearningPreparationSourceInspection::Unavailable {
                reason: "subtitle_track_unavailable".into()
            }
        );
    }
}

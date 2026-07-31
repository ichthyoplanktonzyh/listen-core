use std::sync::Arc;

use application::{
    FoundationPreparationChildRef, LearningPreparationRunId, MediaLearningPreparationCommand,
    MediaLearningPreparationInspector, MediaLearningPreparationRepository,
    MediaLearningPreparationRequest, MediaLearningPreparationSelectionRequired,
    MediaLearningPreparationSourceInspection, MediaLearningPreparationStatus,
    MediaLearningPreparationTarget, MediaLearningPreparationTransition,
    MediaLearningPreparationUseCases, PrepareMediaLearningResult, SubtitleTextTrackSlot,
    SubtitleTextTrackSnapshot,
};
use domain::{LanguageCode, MediaId, SubtitleTrackId};

use crate::SqliteRepository;

struct ExistingSubtitleInspector;

impl MediaLearningPreparationInspector for ExistingSubtitleInspector {
    fn inspect(
        &self,
        target: &MediaLearningPreparationTarget,
        request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, application::ApplicationError> {
        let track_id = request
            .explicit_subtitle_track_id
            .clone()
            .unwrap_or_else(|| SubtitleTrackId::parse("track").unwrap());
        Ok(MediaLearningPreparationSourceInspection::Existing {
            snapshot: SubtitleTextTrackSnapshot {
                media_id: target.media_id.clone(),
                track_id,
                track_fingerprint: format!("raw-{}", target.media_fingerprint),
                text_snapshot_fingerprint: format!("text-{}", target.media_fingerprint),
                language: target
                    .requested_learning_language
                    .clone()
                    .unwrap_or_else(|| LanguageCode::parse("en").unwrap()),
            },
        })
    }
}

struct AsrInspector;

impl MediaLearningPreparationInspector for AsrInspector {
    fn inspect(
        &self,
        _target: &MediaLearningPreparationTarget,
        request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, application::ApplicationError> {
        Ok(MediaLearningPreparationSourceInspection::Asr {
            audio_track: request.explicit_audio_track,
        })
    }
}

struct SelectionInspector;

impl MediaLearningPreparationInspector for SelectionInspector {
    fn inspect(
        &self,
        _target: &MediaLearningPreparationTarget,
        _request: &MediaLearningPreparationRequest,
    ) -> Result<MediaLearningPreparationSourceInspection, application::ApplicationError> {
        Ok(
            MediaLearningPreparationSourceInspection::SelectionRequired {
                reason: MediaLearningPreparationSelectionRequired::AudioTrack,
            },
        )
    }
}

fn target(media_fingerprint: &str) -> MediaLearningPreparationTarget {
    MediaLearningPreparationTarget {
        media_id: MediaId::parse("media").unwrap(),
        media_fingerprint: media_fingerprint.into(),
        requested_learning_language: Some(LanguageCode::parse("en").unwrap()),
    }
}

fn request() -> MediaLearningPreparationRequest {
    MediaLearningPreparationRequest {
        explicit_subtitle_track_id: None,
        explicit_audio_track: None,
    }
}

fn prepared(
    use_cases: &MediaLearningPreparationUseCases,
    media_fingerprint: &str,
    now_ms: u64,
) -> application::MediaLearningPreparation {
    match use_cases
        .prepare(target(media_fingerprint), request(), now_ms)
        .unwrap()
    {
        PrepareMediaLearningResult::Run(run) | PrepareMediaLearningResult::Replaced { run, .. } => {
            *run
        }
        other => panic!("expected preparation run, got {other:?}"),
    }
}

#[test]
fn sqlite_is_media_level_single_flight_authority() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases = Arc::new(MediaLearningPreparationUseCases::new(
        repository.clone(),
        Arc::new(ExistingSubtitleInspector),
    ));

    let handles = (0..8)
        .map(|index| {
            let use_cases = use_cases.clone();
            std::thread::spawn(move || prepared(&use_cases, "media-fp", 100 + index))
        })
        .collect::<Vec<_>>();
    let runs = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();

    assert!(runs.windows(2).all(|pair| pair[0].id == pair[1].id));
    let count: u64 = repository
        .connection
        .lock()
        .query_row(
            "SELECT COUNT(*) FROM media_learning_preparations",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn changed_media_snapshot_invalidates_active_parent_before_replacement() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases = MediaLearningPreparationUseCases::new(
        repository.clone(),
        Arc::new(ExistingSubtitleInspector),
    );
    let first = prepared(&use_cases, "v1", 100);

    let second = match use_cases.prepare(target("v2"), request(), 200).unwrap() {
        PrepareMediaLearningResult::Replaced { run, invalidated } => {
            assert_eq!(invalidated.id, first.id);
            *run
        }
        other => panic!("expected replacement, got {other:?}"),
    };

    assert_ne!(first.id, second.id);
    let invalidated = repository.get(&first.id).unwrap().unwrap();
    assert_eq!(invalidated.status, MediaLearningPreparationStatus::Failed);
    assert_eq!(
        invalidated.error.as_deref(),
        Some("media preparation inputs changed")
    );
}

#[test]
fn retry_keeps_frozen_subtitle_and_records_lineage() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases =
        MediaLearningPreparationUseCases::new(repository, Arc::new(ExistingSubtitleInspector));
    let run = prepared(&use_cases, "media-fp", 100);
    use_cases
        .command(&run.id, MediaLearningPreparationCommand::Start, 110)
        .unwrap();
    use_cases
        .command(
            &run.id,
            MediaLearningPreparationCommand::AcceptExistingSubtitle,
            120,
        )
        .unwrap();
    let foundation_id = LearningPreparationRunId::parse("foundation").unwrap();
    use_cases
        .command(
            &run.id,
            MediaLearningPreparationCommand::AttachFoundationChild {
                child: FoundationPreparationChildRef {
                    run_id: foundation_id.clone(),
                    input_fingerprint: "foundation-input".into(),
                },
            },
            130,
        )
        .unwrap();
    use_cases
        .command(
            &run.id,
            MediaLearningPreparationCommand::FailFoundationChild {
                run_id: foundation_id,
                reason: "foundation failed".into(),
            },
            140,
        )
        .unwrap();

    let retry = use_cases.retry(&run.id, 200).unwrap();

    assert_eq!(retry.retry_of_id.as_ref(), Some(&run.id));
    assert_eq!(retry.status, MediaLearningPreparationStatus::Queued);
    assert!(matches!(
        retry.subtitle_text_track,
        SubtitleTextTrackSlot::Ready { .. }
    ));
}

#[test]
fn startup_recovery_requeues_running_parent_and_keeps_asr_child_provenance() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases = MediaLearningPreparationUseCases::new(repository, Arc::new(AsrInspector));
    let run = prepared(&use_cases, "media-fp", 100);
    use_cases
        .command(&run.id, MediaLearningPreparationCommand::Start, 110)
        .unwrap();
    use_cases
        .command(
            &run.id,
            MediaLearningPreparationCommand::AttachAsrChild {
                job_id: domain::TranscriptionJobId::parse("asr").unwrap(),
                input_provenance_fingerprint: "exact-asr-input".into(),
            },
            120,
        )
        .unwrap();

    let recovered = use_cases.recover_startup(200).unwrap();

    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].status, MediaLearningPreparationStatus::Queued);
    assert!(matches!(
        recovered[0].subtitle_text_track,
        SubtitleTextTrackSlot::AsrChild {
            input_provenance_fingerprint: Some(ref fingerprint),
            ..
        } if fingerprint == "exact-asr-input"
    ));
}

#[test]
fn revision_cas_rejects_stale_writer_and_cancellation_is_durable() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases = MediaLearningPreparationUseCases::new(
        repository.clone(),
        Arc::new(ExistingSubtitleInspector),
    );
    let run = prepared(&use_cases, "media-fp", 100);
    let mut winner = run.clone();
    winner
        .apply(MediaLearningPreparationCommand::Start, 110)
        .unwrap();
    assert!(matches!(
        repository.transition(run.revision, &winner).unwrap(),
        MediaLearningPreparationTransition::Applied(_)
    ));
    let mut stale = run.clone();
    stale
        .apply(MediaLearningPreparationCommand::RequestCancel, 120)
        .unwrap();
    assert!(matches!(
        repository.transition(run.revision, &stale).unwrap(),
        MediaLearningPreparationTransition::Rejected(current)
            if current.status == MediaLearningPreparationStatus::Running
    ));

    let cancelling = use_cases
        .command(&run.id, MediaLearningPreparationCommand::RequestCancel, 130)
        .unwrap();
    assert_eq!(
        cancelling.status,
        MediaLearningPreparationStatus::Cancelling
    );
    let cancelled = use_cases
        .command(
            &run.id,
            MediaLearningPreparationCommand::FinishCancellation,
            140,
        )
        .unwrap();
    assert_eq!(repository.get(&run.id).unwrap(), Some(cancelled.clone()));
    assert_eq!(cancelled.status, MediaLearningPreparationStatus::Cancelled);
}

#[test]
fn unresolved_content_selection_does_not_create_a_parent_run() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases =
        MediaLearningPreparationUseCases::new(repository.clone(), Arc::new(SelectionInspector));

    assert_eq!(
        use_cases
            .prepare(target("media-fp"), request(), 100)
            .unwrap(),
        PrepareMediaLearningResult::SelectionRequired(
            MediaLearningPreparationSelectionRequired::AudioTrack
        )
    );
    let count: u64 = repository
        .connection
        .lock()
        .query_row(
            "SELECT COUNT(*) FROM media_learning_preparations",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn async_coordinator_can_submit_resolved_audio_without_changing_the_intent_request() {
    let repository = Arc::new(SqliteRepository::in_memory().unwrap());
    let use_cases = MediaLearningPreparationUseCases::new(repository, Arc::new(SelectionInspector));
    let intent = request();

    let run = match use_cases
        .prepare_resolved(
            target("media-fp"),
            intent.clone(),
            MediaLearningPreparationSourceInspection::Asr {
                audio_track: Some(3),
            },
            100,
        )
        .unwrap()
    {
        PrepareMediaLearningResult::Run(run) => *run,
        other => panic!("expected preparation run, got {other:?}"),
    };

    assert_eq!(run.request, intent);
    assert!(matches!(
        run.subtitle_text_track,
        SubtitleTextTrackSlot::AsrChild {
            audio_track: Some(3),
            ..
        }
    ));
}

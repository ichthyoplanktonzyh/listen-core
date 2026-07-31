use application::{
    ApplicationError, CreateMediaLearningPreparation, MediaLearningPreparation,
    MediaLearningPreparationId, MediaLearningPreparationRepository, MediaLearningPreparationStatus,
    MediaLearningPreparationTransition,
};
use rusqlite::{OptionalExtension, params};

use crate::{SqliteRepository, from_json, json, repo};

const PREPARATION_COLUMNS: &str = "id,target_key,input_fingerprint,status,revision,retry_of_id,created_at_ms,updated_at_ms,preparation_json";

impl MediaLearningPreparationRepository for SqliteRepository {
    fn create_active(
        &self,
        preparation: &MediaLearningPreparation,
    ) -> Result<CreateMediaLearningPreparation, ApplicationError> {
        validate_identity(preparation)?;
        if !preparation.status.is_active() {
            return Err(ApplicationError::Invalid(
                "a newly created media learning preparation must be active".into(),
            ));
        }
        let mut connection = self.connection.lock();
        let tx = connection.transaction().map_err(repo)?;
        let existing = select_active(&tx, &preparation.target_key)?;
        if let Some(existing) = existing {
            tx.commit().map_err(repo)?;
            return Ok(
                if existing.input_fingerprint == preparation.input_fingerprint {
                    CreateMediaLearningPreparation::Existing(existing)
                } else {
                    CreateMediaLearningPreparation::InputChanged(existing)
                },
            );
        }
        tx.execute(
            "INSERT INTO media_learning_preparations
             (id,target_key,input_fingerprint,status,revision,retry_of_id,
              created_at_ms,updated_at_ms,preparation_json)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                preparation.id.as_str(),
                preparation.target_key,
                preparation.input_fingerprint,
                status(preparation.status),
                preparation.revision,
                preparation.retry_of_id.as_ref().map(|id| id.as_str()),
                preparation.created_at_ms,
                preparation.updated_at_ms,
                json(preparation)?,
            ],
        )
        .map_err(repo)?;
        tx.commit().map_err(repo)?;
        Ok(CreateMediaLearningPreparation::Created(preparation.clone()))
    }

    fn get(
        &self,
        id: &MediaLearningPreparationId,
    ) -> Result<Option<MediaLearningPreparation>, ApplicationError> {
        self.connection
            .lock()
            .query_row(
                &format!(
                    "SELECT {PREPARATION_COLUMNS}
                     FROM media_learning_preparations WHERE id=?1"
                ),
                [id.as_str()],
                read_preparation,
            )
            .optional()
            .map_err(repo)
    }

    fn transition(
        &self,
        expected_revision: u64,
        preparation: &MediaLearningPreparation,
    ) -> Result<MediaLearningPreparationTransition, ApplicationError> {
        validate_identity(preparation)?;
        let connection = self.connection.lock();
        let changed = connection
            .execute(
                "UPDATE media_learning_preparations
                 SET status=?1,revision=?2,updated_at_ms=?3,preparation_json=?4
                 WHERE id=?5 AND revision=?6
                   AND target_key=?7 AND input_fingerprint=?8",
                params![
                    status(preparation.status),
                    preparation.revision,
                    preparation.updated_at_ms,
                    json(preparation)?,
                    preparation.id.as_str(),
                    expected_revision,
                    preparation.target_key,
                    preparation.input_fingerprint,
                ],
            )
            .map_err(repo)?;
        if changed == 1 {
            return Ok(MediaLearningPreparationTransition::Applied(
                preparation.clone(),
            ));
        }
        let current = connection
            .query_row(
                &format!(
                    "SELECT {PREPARATION_COLUMNS}
                     FROM media_learning_preparations WHERE id=?1"
                ),
                [preparation.id.as_str()],
                read_preparation,
            )
            .optional()
            .map_err(repo)?
            .ok_or(ApplicationError::NotFound("media learning preparation"))?;
        Ok(MediaLearningPreparationTransition::Rejected(current))
    }

    fn recover_active(
        &self,
        now_ms: u64,
    ) -> Result<Vec<MediaLearningPreparation>, ApplicationError> {
        let mut connection = self.connection.lock();
        let tx = connection.transaction().map_err(repo)?;
        let mut active = {
            let mut statement = tx
                .prepare(&format!(
                    "SELECT {PREPARATION_COLUMNS}
                     FROM media_learning_preparations
                     WHERE status IN ('queued','running','cancelling')
                     ORDER BY created_at_ms,id"
                ))
                .map_err(repo)?;
            statement
                .query_map([], read_preparation)
                .map_err(repo)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(repo)?
        };
        for preparation in &mut active {
            let expected_revision = preparation.revision;
            preparation.recover_after_restart(now_ms);
            if preparation.revision != expected_revision {
                let changed = tx
                    .execute(
                        "UPDATE media_learning_preparations
                         SET status=?1,revision=?2,updated_at_ms=?3,preparation_json=?4
                         WHERE id=?5 AND revision=?6",
                        params![
                            status(preparation.status),
                            preparation.revision,
                            preparation.updated_at_ms,
                            json(preparation)?,
                            preparation.id.as_str(),
                            expected_revision,
                        ],
                    )
                    .map_err(repo)?;
                if changed != 1 {
                    return Err(ApplicationError::Conflict(
                        "media learning preparation recovery changed concurrently",
                    ));
                }
            }
        }
        tx.commit().map_err(repo)?;
        Ok(active)
    }
}

fn select_active(
    connection: &rusqlite::Connection,
    target_key: &str,
) -> Result<Option<MediaLearningPreparation>, ApplicationError> {
    connection
        .query_row(
            &format!(
                "SELECT {PREPARATION_COLUMNS}
                 FROM media_learning_preparations
                 WHERE target_key=?1 AND status IN ('queued','running','cancelling')"
            ),
            [target_key],
            read_preparation,
        )
        .optional()
        .map_err(repo)
}

fn read_preparation(row: &rusqlite::Row<'_>) -> rusqlite::Result<MediaLearningPreparation> {
    let stored_id: String = row.get(0)?;
    let stored_target_key: String = row.get(1)?;
    let stored_input_fingerprint: String = row.get(2)?;
    let stored_status: String = row.get(3)?;
    let stored_revision: u64 = row.get(4)?;
    let stored_retry_of_id: Option<String> = row.get(5)?;
    let stored_created_at_ms: u64 = row.get(6)?;
    let stored_updated_at_ms: u64 = row.get(7)?;
    let preparation: MediaLearningPreparation = from_json(&row.get::<_, String>(8)?)?;
    if preparation.id.as_str() != stored_id
        || preparation.target_key != stored_target_key
        || preparation.input_fingerprint != stored_input_fingerprint
        || status(preparation.status) != stored_status
        || preparation.revision != stored_revision
        || preparation.retry_of_id.as_ref().map(|id| id.as_str()) != stored_retry_of_id.as_deref()
        || preparation.created_at_ms != stored_created_at_ms
        || preparation.updated_at_ms != stored_updated_at_ms
        || !preparation.has_valid_identity()
    {
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(preparation)
}

fn validate_identity(preparation: &MediaLearningPreparation) -> Result<(), ApplicationError> {
    if !preparation.has_valid_identity() {
        return Err(ApplicationError::Invalid(
            "media learning preparation identity changed".into(),
        ));
    }
    Ok(())
}

fn status(value: MediaLearningPreparationStatus) -> &'static str {
    match value {
        MediaLearningPreparationStatus::Queued => "queued",
        MediaLearningPreparationStatus::Running => "running",
        MediaLearningPreparationStatus::Cancelling => "cancelling",
        MediaLearningPreparationStatus::Completed => "completed",
        MediaLearningPreparationStatus::Failed => "failed",
        MediaLearningPreparationStatus::Cancelled => "cancelled",
    }
}

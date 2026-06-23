"""
Initial schema — video_records, detection_results, embedding_records.

The ORM source of truth is the shared models:
- shared.models.video.VideoRecord
- shared.models.detection_result.DetectionResult
- shared.models.embedding_record.EmbeddingRecord

This migration must stay in sync with those models.

Run:
    alembic -c infra/migrations/alembic.ini upgrade head
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # video_records
    # -------------------------------------------------------------------------
    op.create_table(
        "video_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256_hash", sa.String(64), nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=True),
        sa.Column("storage_bucket", sa.String(256), nullable=True),
        sa.Column("camera_id", sa.String(256), nullable=True),
        sa.Column("location", sa.String(512), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("resolution_width", sa.Integer, nullable=True),
        sa.Column("resolution_height", sa.Integer, nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "PROCESSING",
                "PREPROCESSED",
                "DETECTED",
                "EMBEDDED",
                "INDEXED",
                "FAILED",
                "QUARANTINED",
                "DUPLICATE",
                name="videostatus",
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_unique_constraint("uq_video_hash", "video_records", ["sha256_hash"])
    op.create_index("ix_video_status", "video_records", ["status"])
    op.create_index("ix_video_camera_id", "video_records", ["camera_id"])
    op.create_index("ix_video_created_at", "video_records", ["created_at"])

    # -------------------------------------------------------------------------
    # detection_results
    # -------------------------------------------------------------------------
    op.create_table(
        "detection_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_records.id"),
            nullable=False,
        ),
        sa.Column("frame_path", sa.String(1024), nullable=False),
        sa.Column("frame_timestamp_ms", sa.Integer, nullable=False),
        sa.Column("scene_id", sa.Integer, nullable=False, server_default="0"),
        sa.Column("label", sa.String(512), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("bbox_x1", sa.Float, nullable=False),
        sa.Column("bbox_y1", sa.Float, nullable=False),
        sa.Column("bbox_x2", sa.Float, nullable=False),
        sa.Column("bbox_y2", sa.Float, nullable=False),
        sa.Column("crop_path", sa.String(1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_det_video_id", "detection_results", ["video_id"])
    op.create_index("ix_det_frame_ts", "detection_results", ["frame_timestamp_ms"])
    op.create_index("ix_det_scene_id", "detection_results", ["scene_id"])

    # -------------------------------------------------------------------------
    # embedding_records
    # -------------------------------------------------------------------------
    op.create_table(
        "embedding_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_records.id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("source_path", sa.String(1024), nullable=False),
        sa.Column("model_name", sa.String(256), nullable=False),
        sa.Column("timestamp_ms", sa.Integer, nullable=True),
        sa.Column("label", sa.String(512), nullable=True),
        sa.Column("qdrant_point_id", sa.String(64), nullable=True),
        sa.Column("qdrant_collection", sa.String(256), nullable=True),
        sa.Column("vector_dim", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_emb_video_id", "embedding_records", ["video_id"])
    op.create_index("ix_emb_kind", "embedding_records", ["kind"])
    op.create_index("ix_emb_qdrant_id", "embedding_records", ["qdrant_point_id"])


def downgrade() -> None:
    op.drop_index("ix_emb_qdrant_id", table_name="embedding_records")
    op.drop_index("ix_emb_kind", table_name="embedding_records")
    op.drop_index("ix_emb_video_id", table_name="embedding_records")
    op.drop_table("embedding_records")

    op.drop_index("ix_det_scene_id", table_name="detection_results")
    op.drop_index("ix_det_frame_ts", table_name="detection_results")
    op.drop_index("ix_det_video_id", table_name="detection_results")
    op.drop_table("detection_results")

    op.drop_index("ix_video_created_at", table_name="video_records")
    op.drop_index("ix_video_camera_id", table_name="video_records")
    op.drop_index("ix_video_status", table_name="video_records")
    op.drop_constraint("uq_video_hash", "video_records", type_="unique")
    op.drop_table("video_records")

    op.execute("DROP TYPE IF EXISTS videostatus")
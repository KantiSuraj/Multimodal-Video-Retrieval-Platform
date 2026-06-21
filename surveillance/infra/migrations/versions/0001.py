"""
Initial schema — video_records table.

The ORM source of truth is shared.models.video.VideoRecord.
This migration must stay in sync with that model.

Run:  alembic -c infra/migrations/alembic.ini upgrade head
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision       = "0001_initial"
down_revision  = None
branch_labels  = None
depends_on     = None


def upgrade() -> None:
    op.create_table(
        "video_records",
        sa.Column("id",                postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256_hash",       sa.String(64),   nullable=False),
        sa.Column("original_filename", sa.String(512),  nullable=False),
        sa.Column("mime_type",         sa.String(128),  nullable=False),
        sa.Column("file_size_bytes",   sa.BigInteger,   nullable=False),
        sa.Column("storage_path",      sa.String(1024), nullable=True),
        sa.Column("storage_bucket",    sa.String(256),  nullable=True),
        sa.Column("camera_id",         sa.String(256),  nullable=True),
        sa.Column("location",          sa.String(512),  nullable=True),
        sa.Column("recorded_at",       sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds",  sa.Float,    nullable=True),
        sa.Column("resolution_width",  sa.Integer,  nullable=True),
        sa.Column("resolution_height", sa.Integer,  nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "PROCESSING","PREPROCESSED", "INDEXED",
                "FAILED", "QUARANTINED", "DUPLICATE",
                name="videostatus",
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at",    sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at",    sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_video_hash",    "video_records", ["sha256_hash"])
    op.create_index("ix_video_status",    "video_records", ["status"])
    op.create_index("ix_video_camera_id", "video_records", ["camera_id"])
    op.create_index("ix_video_created_at","video_records", ["created_at"])


def downgrade() -> None:
    op.drop_table("video_records")
    op.execute("DROP TYPE IF EXISTS videostatus")
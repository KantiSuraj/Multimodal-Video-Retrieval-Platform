"""
Add composite indexes on embedding_records for Search Service hydration.

The Search Service hydrates each Qdrant result point with PostgreSQL
metadata via:

    SELECT * FROM embedding_records
    WHERE video_id = :video_id AND source_path = :source_path

Without a composite index this is a full table scan on a potentially
multi-million-row table.  The index on (video_id, source_path) makes
this lookup O(log n).

Also adds:
  - (video_id, kind) — for future kind-filtered hydration queries
  - (video_id, timestamp_ms) — for temporal range queries on search results

These indexes are pure additions; no schema columns change.
Migration 0001 remains the authoritative schema creation.

Run:
    alembic -c infra/migrations/alembic.ini upgrade head
"""
from alembic import op

revision = "0002_search_hydration_indexes"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Critical for Search hydration: WHERE video_id = X AND source_path = Y
    op.create_index(
        "ix_emb_video_source",
        "embedding_records",
        ["video_id", "source_path"],
    )

    # For kind-filtered queries (e.g. "only crop embeddings")
    op.create_index(
        "ix_emb_video_kind",
        "embedding_records",
        ["video_id", "kind"],
    )

    # For temporal range queries on search result sets
    op.create_index(
        "ix_emb_video_ts",
        "embedding_records",
        ["video_id", "timestamp_ms"],
    )


def downgrade() -> None:
    op.drop_index("ix_emb_video_ts", table_name="embedding_records")
    op.drop_index("ix_emb_video_kind", table_name="embedding_records")
    op.drop_index("ix_emb_video_source", table_name="embedding_records")

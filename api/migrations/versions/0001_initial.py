"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-29

Соответствует моделям в models/models.py (включая researches.error_message).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("password", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_table(
        "statuses",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status_name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "localizations",
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("zone_name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("zone_id"),
    )
    op.create_table(
        "classifications",
        sa.Column("classification_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("class_name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("classification_id"),
    )
    op.create_table(
        "researches",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("upload_date", sa.TIMESTAMP(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("status_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.ForeignKeyConstraint(["status_id"], ["statuses.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "classes_areas",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("localization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("area", sa.Float(), nullable=True),
        sa.Column("class_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("research_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["localization_id"], ["localizations.zone_id"]),
        sa.ForeignKeyConstraint(["class_id"], ["classifications.classification_id"]),
        sa.ForeignKeyConstraint(["research_id"], ["researches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("classes_areas")
    op.drop_table("researches")
    op.drop_table("classifications")
    op.drop_table("localizations")
    op.drop_table("statuses")
    op.drop_table("patients")
    op.drop_table("users")

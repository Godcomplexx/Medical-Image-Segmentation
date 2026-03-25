import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.models import Status as StatusModel

async def get_status_by_id(db: AsyncSession, status_id: uuid.UUID) -> StatusModel | None:
    result = await db.execute(select(StatusModel).where(StatusModel.id == status_id))
    return result.scalars().first()

async def get_status_by_name(db: AsyncSession, status_name: str) -> StatusModel | None:
    result = await db.execute(select(StatusModel).where(StatusModel.status_name == status_name))
    return result.scalars().first()
# patch (AAK): data-pruning endpoint for the ITK AI Platform (Aarhus Kommune).
#
# Upstream open-webui has no data-retention mechanism, so chats accumulate
# indefinitely. This router adds an admin-only endpoint that deletes chats
# not updated for a given number of days, intended to be called from a cron
# job. The endpoint lives in its own file so the patch only touches main.py
# for import and registration.

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query
from open_webui.internal.db import get_async_session
from open_webui.models.chats import Chat, Chats
from open_webui.utils.auth import get_admin_user
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

router = APIRouter()


class PruneChatsResponse(BaseModel):
    dry_run: bool
    days: int
    cutoff: int
    matched: int
    deleted: int
    failed: int


@router.post('/chats', response_model=PruneChatsResponse)
async def prune_chats(
    days: int = Query(..., ge=1, description='Delete chats not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    limit: Optional[int] = Query(None, ge=1, description='Cap the number of chats deleted in this call'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    # Chats last updated before this epoch timestamp are pruned.
    cutoff = int(time.time()) - days * 24 * 60 * 60

    # Pinned chats are deliberately spared; archived chats are pruned by age
    # like any other. Oldest first, so a limited run works through the backlog.
    query = (
        select(Chat.id, Chat.user_id, Chat.meta)
        .where(Chat.updated_at < cutoff)
        .where(Chat.pinned.isnot(True))
        .order_by(Chat.updated_at.asc())
    )
    if limit is not None:
        query = query.limit(limit)

    rows = (await db.execute(query)).all()

    matched = len(rows)
    deleted = 0
    failed = 0

    if not dry_run:
        # Track which tags (meta['tags']) the deleted chats used, per user,
        # so tags no longer used by any chat can be removed afterwards.
        tag_ids_by_user: dict[str, set[str]] = {}
        for row in rows:
            # One transaction per chat: a failure mid-run leaves already
            # deleted chats gone and reports the rest as failed.
            if await Chats.delete_chat_by_id(row.id):
                deleted += 1
                tag_ids_by_user.setdefault(row.user_id, set()).update((row.meta or {}).get('tags', []))
            else:
                failed += 1

        for user_id, tag_ids in tag_ids_by_user.items():
            if tag_ids:
                # We already deleted the chats, so an orphaned tag has zero
                # remaining references, so we use threshold=0.
                await Chats.delete_orphan_tags_for_user(list(tag_ids), user_id, threshold=0, db=db)

        log.info(
            f'prune_chats by {user.email}: days={days} cutoff={cutoff} '
            f'matched={matched} deleted={deleted} failed={failed}'
        )

    return PruneChatsResponse(
        dry_run=dry_run,
        days=days,
        cutoff=cutoff,
        matched=matched,
        deleted=deleted,
        failed=failed,
    )

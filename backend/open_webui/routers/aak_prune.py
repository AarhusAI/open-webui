# patch (AAK): data-pruning endpoints for the ITK AI Platform (Aarhus Kommune).
#
# Upstream open-webui has no data-retention mechanism, so user data accumulates
# indefinitely. The retention policy requires removing data not updated for N
# days. This router adds admin-only endpoints, intended to be called from a
# cron job:
#   POST /chats            - chats (incl. pinned/archived), their messages,
#                            orphaned tags and the files uploaded in them
#   POST /memories         - memory rows and their user-memory vectors
#   POST /feedbacks        - ratings; each stores a full chat copy in `snapshot`
#   POST /channel-messages - channel messages and their reactions
#   POST /all              - runs all of the above, so cron needs one call
# The endpoints live in their own file so the patch only touches main.py for
# import and registration.

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query
from open_webui.internal.db import get_async_session
from open_webui.models.channels import ChannelFile
from open_webui.models.chats import Chat, ChatFile, Chats
from open_webui.models.feedbacks import Feedback
from open_webui.models.files import Files
from open_webui.models.knowledge import KnowledgeFile
from open_webui.models.memories import Memory
from open_webui.models.messages import Message, MessageReaction
from open_webui.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from open_webui.routers.knowledge import delete_file_resource
from open_webui.utils.auth import get_admin_user
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

router = APIRouter()


class PruneResponse(BaseModel):
    dry_run: bool
    days: int
    cutoff: int
    matched: int
    deleted: int
    failed: int


class PruneChatsResponse(PruneResponse):
    files_deleted: int = 0
    files_failed: int = 0

class PruneAllResponse(BaseModel):
    chats: PruneChatsResponse
    memories: PruneResponse
    feedbacks: PruneResponse
    channel_messages: PruneResponse

def _cutoff(days: int) -> int:
    # Rows last updated before this epoch timestamp (seconds) are pruned.
    return int(time.time()) - days * 24 * 60 * 60


async def _file_in_use(file_id: str, db: AsyncSession) -> bool:
    # Join with Chat so links of chats deleted in this run never count,
    # regardless of whether the FK cascade removed them.
    queries = (
        select(ChatFile.id).join(Chat, Chat.id == ChatFile.chat_id).where(ChatFile.file_id == file_id).limit(1),
        select(KnowledgeFile.id).where(KnowledgeFile.file_id == file_id).limit(1),
        select(ChannelFile.id).where(ChannelFile.file_id == file_id).limit(1),
    )
    for q in queries:
        if (await db.execute(q)).first() is not None:
            return True
    return False

@router.post('/all', response_model=PruneAllResponse)
async def prune_all(
    days: int = Query(..., ge=1, description='Delete data not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    # Called directly, so every argument is passed explicitly (the defaults
    # are FastAPI Query objects).
    # noinspection PyArgumentList
    return PruneAllResponse(
        chats=await prune_chats(days=days, dry_run=dry_run, limit=None, user=user, db=db),
        memories=await prune_memories(days=days, dry_run=dry_run, user=user, db=db),
        feedbacks=await prune_feedbacks(days=days, dry_run=dry_run, user=user, db=db),
        channel_messages=await prune_channel_messages(days=days, dry_run=dry_run, user=user, db=db),
    )


@router.post('/chats', response_model=PruneChatsResponse)
async def prune_chats(
    days: int = Query(..., ge=1, description='Delete chats not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    limit: Optional[int] = Query(None, ge=1, description='Cap the number of chats deleted in this call'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)

    # All chats, including pinned and archived, are pruned by age (retention
    # policy has no exceptions). Oldest first, so a limited run works through
    # the backlog.
    query = select(Chat.id, Chat.user_id, Chat.meta).where(Chat.updated_at < cutoff).order_by(Chat.updated_at.asc())
    if limit is not None:
        query = query.limit(limit)

    rows = (await db.execute(query)).all()

    matched = len(rows)
    deleted = 0
    failed = 0
    files_deleted = 0
    files_failed = 0

    if not dry_run:
        # Track which tags (meta['tags']) the deleted chats used, per user,
        # so tags no longer used by any chat can be removed afterwards.
        tag_ids_by_user: dict[str, set[str]] = {}
        # file_id -> owner of the deleted chat. chat_file rows cascade away
        # with the chat, so file ids must be read before the delete.
        candidate_files: dict[str, str] = {}
        for row in rows:
            file_ids = (
                (await db.execute(select(ChatFile.file_id).where(ChatFile.chat_id == row.id))).scalars().all()
            )
            # One transaction per chat: a failure mid-run leaves already
            # deleted chats gone and reports the rest as failed.
            if await Chats.delete_chat_by_id(row.id):
                deleted += 1
                tag_ids_by_user.setdefault(row.user_id, set()).update((row.meta or {}).get('tags', []))
                for file_id in file_ids:
                    candidate_files[file_id] = row.user_id
            else:
                failed += 1

        for user_id, tag_ids in tag_ids_by_user.items():
            if tag_ids:
                # We already deleted the chats, so an orphaned tag has zero
                # remaining references, so we use threshold=0.
                await Chats.delete_orphan_tags_for_user(list(tag_ids), user_id, threshold=0, db=db)

        for file_id, owner_id in candidate_files.items():
            file = await Files.get_file_by_id(file_id, db=db)
            # Only delete files the chat owner uploaded and nothing else uses
            # (other chats, knowledge bases, channels).
            if file is None or file.user_id != owner_id or await _file_in_use(file_id, db):
                continue
            if await delete_file_resource(file, db):
                files_deleted += 1
            else:
                files_failed += 1

        log.info(
            f'prune_chats by {user.email}: days={days} cutoff={cutoff} '
            f'matched={matched} deleted={deleted} failed={failed} '
            f'files_deleted={files_deleted} files_failed={files_failed}'
        )

    return PruneChatsResponse(
        dry_run=dry_run,
        days=days,
        cutoff=cutoff,
        matched=matched,
        deleted=deleted,
        failed=failed,
        files_deleted=files_deleted,
        files_failed=files_failed,
    )


@router.post('/memories', response_model=PruneResponse)
async def prune_memories(
    days: int = Query(..., ge=1, description='Delete memories not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)
    rows = (await db.execute(select(Memory.id, Memory.user_id).where(Memory.updated_at < cutoff))).all()
    matched = len(rows)
    deleted = 0
    failed = 0

    if not dry_run:
        ids_by_user: dict[str, list[str]] = {}
        for row in rows:
            ids_by_user.setdefault(row.user_id, []).append(row.id)

        for uid, ids in ids_by_user.items():
            # Vectors first: rows deleted while vectors remain would keep the
            # memory retrievable. On failure the rows stay so the next run retries.
            try:
                await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{uid}', ids=ids)
            except Exception:
                log.exception(f'prune_memories: vector delete failed for user {uid}')
                failed += len(ids)
                continue
            res = await db.execute(delete(Memory).where(Memory.user_id == uid, Memory.id.in_(ids)))
            await db.commit()
            deleted += res.rowcount

        log.info(
            f'prune_memories by {user.email}: days={days} cutoff={cutoff} '
            f'matched={matched} deleted={deleted} failed={failed}'
        )

    return PruneResponse(dry_run=dry_run, days=days, cutoff=cutoff, matched=matched, deleted=deleted, failed=failed)


@router.post('/feedbacks', response_model=PruneResponse)
async def prune_feedbacks(
    days: int = Query(..., ge=1, description='Delete feedbacks not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    # Each rating stores a full chat copy in feedback.snapshot, so it survives
    # chat deletion. Pruning also removes old ratings from the leaderboard.
    cutoff = _cutoff(days)
    where = Feedback.updated_at < cutoff
    matched = (await db.execute(select(func.count()).select_from(Feedback).where(where))).scalar()
    deleted = 0

    if not dry_run:
        res = await db.execute(delete(Feedback).where(where))
        await db.commit()
        deleted = res.rowcount

        log.info(
            f'prune_feedbacks by {user.email}: days={days} cutoff={cutoff} '
            f'matched={matched} deleted={deleted} failed=0'
        )

    return PruneResponse(dry_run=dry_run, days=days, cutoff=cutoff, matched=matched, deleted=deleted, failed=0)


@router.post('/channel-messages', response_model=PruneResponse)
async def prune_channel_messages(
    days: int = Query(..., ge=1, description='Delete channel messages not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)
    # Message timestamps are time.time_ns().
    where = Message.updated_at < cutoff * 1_000_000_000
    matched = (await db.execute(select(func.count()).select_from(Message).where(where))).scalar()
    deleted = 0

    if not dry_run:
        # Pinned messages are pruned too (no exceptions). Deletion is per
        # message age, like upstream's single-message delete: newer thread
        # replies outlive their parent and are pruned once they age out.
        # channel_file links cascade by FK; channel files themselves are kept.
        await db.execute(delete(MessageReaction).where(MessageReaction.message_id.in_(select(Message.id).where(where))))
        res = await db.execute(delete(Message).where(where))
        await db.commit()
        deleted = res.rowcount

        log.info(
            f'prune_channel_messages by {user.email}: days={days} cutoff={cutoff} '
            f'matched={matched} deleted={deleted} failed=0'
        )

    return PruneResponse(dry_run=dry_run, days=days, cutoff=cutoff, matched=matched, deleted=deleted, failed=0)

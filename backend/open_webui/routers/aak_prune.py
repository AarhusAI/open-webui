# patch (AAK): data-pruning endpoints for the ITK AI Platform (Aarhus Kommune).
#
# Upstream open-webui has no data-retention mechanism, so user data accumulates
# indefinitely. The retention policy requires removing data not updated for N
# days. This router adds admin-only endpoints, intended to be called from a
# cron job:
#   POST /chats            - chats (incl. pinned/archived), their messages and
#                            orphaned tags
#   POST /memories         - memory rows and their user-memory vectors; not
#                            part of /all, call it separately when wanted
#   POST /feedbacks        - ratings; each stores a full chat copy in `snapshot`
#   POST /channel-messages - channel messages and their reactions
#   POST /files            - files older than N days that nothing references;
#                            knowledge-base files are never touched
#   POST /all              - runs chats, feedbacks, channel-messages and
#                            files (last), so cron needs one call. Memories
#                            are left out by policy; prune them explicitly
#                            via /memories.
# The endpoints live in their own file so the patch only touches main.py for
# import and registration.

from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import APIRouter, Depends, Query
from open_webui.internal.db import get_async_session
from open_webui.models.channels import ChannelFile
from open_webui.models.chats import Chat, ChatFile, Chats
from open_webui.models.feedbacks import Feedback
from open_webui.models.files import File, Files
from open_webui.models.folders import Folder
from open_webui.models.knowledge import Knowledge, KnowledgeFile
from open_webui.models.memories import Memory
from open_webui.models.messages import Message, MessageReaction
from open_webui.models.models import Model
from open_webui.models.notes import Note
from open_webui.models.shared_chats import SharedChat
from open_webui.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from open_webui.storage.provider import Storage
from open_webui.utils.auth import get_admin_user
from pydantic import BaseModel
from sqlalchemy import String, and_, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

router = APIRouter()

# Optional floor for `days`, so a typo in cron or a leaked admin key cannot
# wipe recent data. Unset means no floor beyond 1 day.
MIN_DAYS = int(os.environ.get('AAK_PRUNE_MIN_DAYS', '1'))


class PruneResponse(BaseModel):
    dry_run: bool
    days: int
    cutoff: int
    # Rows selected by the cutoff: what a real run deletes (or tries to).
    # In a dry run this is the answer; `deleted` and `failed` stay 0.
    would_delete: int
    deleted: int
    failed: int


class PruneAllResponse(BaseModel):
    chats: PruneResponse
    feedbacks: PruneResponse
    channel_messages: PruneResponse
    files: PruneResponse


def _cutoff(days: int) -> int:
    # Rows last updated before this epoch timestamp (seconds) are pruned.
    return int(time.time()) - days * 24 * 60 * 60


def _log_run(job: str, user, r: PruneResponse) -> None:
    # Retention spec: one log entry per run with counts only, no content.
    # Goes to stdout via open-webui's loguru setup, which adds the timestamp.
    log.info(f'prune_{job}: admin_id={user.id} ' + ' '.join(f'{k}={v}' for k, v in r.model_dump().items()))


def _unreferenced_file(cutoff: int):
    # A file may be deleted only when it is older than the cutoff AND nothing
    # refers to it. Link tables are not enough: cloned, imported and forked
    # chats, generated images, models, folders, notes and channel messages
    # hold the file id only inside JSON, as a bare id or as
    # /api/v1/files/{id}/content. File ids are server-generated uuids, so the
    # text match can only over-match, which keeps the file.
    # ponytail: the JSON text scans are sequential (seconds on a few thousand
    # files); add a file-reference table if the sweep gets too slow.
    refs = [
        # Join Chat: on SQLite the FK cascade is not enforced, so links of
        # deleted chats may linger.
        select(ChatFile.id).join(Chat, Chat.id == ChatFile.chat_id).where(ChatFile.file_id == File.id),
        select(KnowledgeFile.id).where(KnowledgeFile.file_id == File.id),
        # Uploaded for a knowledge base that still exists but not linked
        # (pending or failed processing): knowledge-base files are never pruned.
        select(Knowledge.id).where(Knowledge.id == File.meta['data']['knowledge_id'].as_string()),
        select(ChannelFile.id).where(ChannelFile.file_id == File.id),
        # ponytail: scans only the file owner's chats to use the user_id index;
        # another user's clone of a shared chat does not keep the file alive
        # (upstream grants that user no access once the source chat is gone).
        select(Chat.id).where(Chat.user_id == File.user_id, cast(Chat.chat, String).contains(File.id)),
    ] + [
        select(col.class_.id).where(cast(col, String).contains(File.id))
        for col in (SharedChat.chat, Message.data, Model.meta, Folder.data, Note.data)
    ]
    # NULL timestamps (legacy rows) count as old.
    return and_(func.coalesce(File.updated_at, File.created_at, 0) < cutoff, *(~q.exists() for q in refs))


async def _delete_file(file_id: str, path: str | None) -> bool:
    # Row last: if vectors or the blob fail, the row stays and the next run
    # retries. Upstream's delete_file_resource swallows those errors and
    # deletes the row anyway, which would orphan the blob for good.
    # ponytail: relies on idempotent blob delete (S3, local). GCS and Azure
    # raise for a missing blob, so such a row would fail on every run.
    try:
        collection = f'file-{file_id}'
        if await ASYNC_VECTOR_DB_CLIENT.has_collection(collection_name=collection):
            await ASYNC_VECTOR_DB_CLIENT.delete_collection(collection_name=collection)
        if path:
            await asyncio.to_thread(Storage.delete_file, path)
        return await Files.delete_file_by_id(file_id)
    except Exception:
        log.exception(f'prune_files: delete failed for file {file_id}')
        return False


# Cron calls /all?limit=N repeatedly until chats.deleted and files.deleted are
# both 0, so no single request is unbounded. Stop on `deleted`, not
# `would_delete`: a row that keeps failing is selected again on every run.
@router.post('/all', response_model=PruneAllResponse)
async def prune_all(
    days: int = Query(..., ge=MIN_DAYS, description='Delete data not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    limit: int | None = Query(None, ge=1, description='Cap the number of chats and of files handled in this call'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    # Called directly, so every argument is passed explicitly (the defaults
    # are FastAPI Query objects). Files run last, so files released by the
    # earlier steps are swept in the same call.
    # noinspection PyArgumentList
    return PruneAllResponse(
        chats=await prune_chats(days=days, dry_run=dry_run, limit=limit, user=user, db=db),
        feedbacks=await prune_feedbacks(days=days, dry_run=dry_run, user=user, db=db),
        channel_messages=await prune_channel_messages(days=days, dry_run=dry_run, user=user, db=db),
        files=await prune_files(days=days, dry_run=dry_run, limit=limit, user=user, db=db),
    )


@router.post('/chats', response_model=PruneResponse)
async def prune_chats(
    days: int = Query(..., ge=MIN_DAYS, description='Delete chats not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    limit: int | None = Query(None, ge=1, description='Cap the number of chats deleted in this call'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)

    # All chats, including pinned and archived, are pruned by age (retention
    # policy has no exceptions). Oldest first, so a limited run works through
    # the backlog. Their files are removed by /files once nothing else
    # references them.
    query = select(Chat.id, Chat.user_id, Chat.meta).where(Chat.updated_at < cutoff).order_by(Chat.updated_at.asc())
    if limit is not None:
        query = query.limit(limit)

    rows = (await db.execute(query)).all()

    would_delete = len(rows)
    deleted = 0
    failed = 0

    if not dry_run:
        # Track which tags (meta['tags']) the deleted chats used, per user,
        # so tags no longer used by any chat can be removed afterwards.
        tag_ids_by_user: dict[str, set[str]] = {}
        for row in rows:
            # Recheck: a chat updated (or deleted) since the scan is skipped
            # and counted neither as deleted nor as failed.
            still_old = select(Chat.id).where(Chat.id == row.id, Chat.updated_at < cutoff)
            if (await db.execute(still_old)).first() is None:
                continue
            # One transaction per chat: a failure mid-run leaves already
            # deleted chats gone and reports the rest as failed.
            if await Chats.delete_chat_by_id(row.id):
                deleted += 1
                tag_ids_by_user.setdefault(row.user_id, set()).update((row.meta or {}).get('tags', []))
            else:
                failed += 1

        for user_id, tag_ids in tag_ids_by_user.items():
            # We already deleted the chats, so an orphaned tag has zero
            # remaining references, so we use threshold=0.
            # Chunked and guarded: upstream counts with one subquery per tag
            # in a single SELECT, and Postgres caps a target list at 1664
            # columns. A failure must not abort the run; leftover tags are
            # empty labels, not user content.
            ids = list(tag_ids)
            for i in range(0, len(ids), 500):
                try:
                    await Chats.delete_orphan_tags_for_user(ids[i : i + 500], user_id, threshold=0, db=db)
                except Exception:
                    log.exception(f'prune_chats: tag cleanup failed for user {user_id}')

    resp = PruneResponse(
        dry_run=dry_run, days=days, cutoff=cutoff, would_delete=would_delete, deleted=deleted, failed=failed
    )
    _log_run('chats', user, resp)
    return resp


@router.post('/memories', response_model=PruneResponse)
async def prune_memories(
    days: int = Query(..., ge=MIN_DAYS, description='Delete memories not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)
    rows = (await db.execute(select(Memory.id, Memory.user_id).where(Memory.updated_at < cutoff))).all()
    would_delete = len(rows)
    deleted = 0
    failed = 0

    if not dry_run:
        ids_by_user: dict[str, list[str]] = {}
        for row in rows:
            ids_by_user.setdefault(row.user_id, []).append(row.id)

        for uid, ids in ids_by_user.items():
            # Vectors first: rows deleted while vectors remain would keep the
            # memory retrievable. On failure the rows stay so the next run retries.
            # Chroma's delete swallows its own errors, so "vectors first" cannot
            # be guaranteed on that backend; the deployment runs Qdrant.
            try:
                name = f'user-memory-{uid}'
                # No collection means no vectors. Plain Qdrant raises on a missing
                # collection, which would otherwise keep these rows failing forever.
                if await ASYNC_VECTOR_DB_CLIENT.has_collection(collection_name=name):
                    await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=name, ids=ids)
            except Exception:
                log.exception(f'prune_memories: vector delete failed for user {uid}')
                failed += len(ids)
                continue
            res = await db.execute(delete(Memory).where(Memory.user_id == uid, Memory.id.in_(ids)))
            await db.commit()
            deleted += res.rowcount

    resp = PruneResponse(
        dry_run=dry_run, days=days, cutoff=cutoff, would_delete=would_delete, deleted=deleted, failed=failed
    )
    _log_run('memories', user, resp)
    return resp


@router.post('/feedbacks', response_model=PruneResponse)
async def prune_feedbacks(
    days: int = Query(..., ge=MIN_DAYS, description='Delete feedbacks not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    # Each rating stores a full chat copy in feedback.snapshot, so it survives
    # chat deletion. Pruning also removes old ratings from the leaderboard.
    cutoff = _cutoff(days)
    where = Feedback.updated_at < cutoff
    would_delete = (await db.execute(select(func.count()).select_from(Feedback).where(where))).scalar()
    deleted = 0

    if not dry_run:
        res = await db.execute(delete(Feedback).where(where))
        await db.commit()
        deleted = res.rowcount

    resp = PruneResponse(
        dry_run=dry_run, days=days, cutoff=cutoff, would_delete=would_delete, deleted=deleted, failed=0
    )
    _log_run('feedbacks', user, resp)
    return resp


@router.post('/channel-messages', response_model=PruneResponse)
async def prune_channel_messages(
    days: int = Query(..., ge=MIN_DAYS, description='Delete channel messages not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)
    # Message timestamps are time.time_ns().
    where = Message.updated_at < cutoff * 1_000_000_000
    would_delete = (await db.execute(select(func.count()).select_from(Message).where(where))).scalar()
    deleted = 0

    if not dry_run:
        # Pinned messages are pruned too (no exceptions). Deletion is per
        # message age, like upstream's single-message delete: newer thread
        # replies outlive their parent and are pruned once they age out.
        # channel_file links cascade by FK; the files are removed by /files.
        await db.execute(delete(MessageReaction).where(MessageReaction.message_id.in_(select(Message.id).where(where))))
        res = await db.execute(delete(Message).where(where))
        await db.commit()
        deleted = res.rowcount

    resp = PruneResponse(
        dry_run=dry_run, days=days, cutoff=cutoff, would_delete=would_delete, deleted=deleted, failed=0
    )
    _log_run('channel_messages', user, resp)
    return resp


# A sweep rather than "delete the files of the chats pruned in this run": it
# also picks up files left by interrupted runs, by chats older than the
# chat_file table and by pruned channel messages, and it retries files whose
# delete failed earlier. A dry run cannot count files that only become
# unreferenced once /chats or /channel-messages really delete.
@router.post('/files', response_model=PruneResponse)
async def prune_files(
    days: int = Query(..., ge=MIN_DAYS, description='Delete unreferenced files not updated for this many days'),
    dry_run: bool = Query(False, description='Only report what would be deleted'),
    limit: int | None = Query(None, ge=1, description='Cap the number of files handled in this call'),
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    cutoff = _cutoff(days)
    where = _unreferenced_file(cutoff)
    query = select(File.id).where(where).order_by(File.created_at.asc())
    if limit is not None:
        query = query.limit(limit)

    ids = (await db.execute(query)).scalars().all()
    would_delete = len(ids)
    deleted = 0
    failed = 0

    if not dry_run:
        for file_id in ids:
            # Recheck: a file attached (or deleted) since the scan is skipped.
            row = (await db.execute(select(File.id, File.path).where(File.id == file_id, where))).first()
            if row is None:
                continue
            if await _delete_file(row.id, row.path):
                deleted += 1
            else:
                failed += 1

    resp = PruneResponse(
        dry_run=dry_run, days=days, cutoff=cutoff, would_delete=would_delete, deleted=deleted, failed=failed
    )
    _log_run('files', user, resp)
    return resp

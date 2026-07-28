# --- BEGIN EXTERNAL RETRIEVAL PATCH ---
# External retrieval engine: delegates document search to an external HTTP service
# instead of querying the built-in vector DB directly.
#
# NOTE: this module is named external_service.py (not external.py) because
# upstream v0.11.0 ships its own retrieval/external.py ("external knowledge":
# per-KB connections straight to Qdrant/Milvus/pgvector). The two features are
# independent and coexist.
# --- END EXTERNAL RETRIEVAL PATCH ---
# --- BEGIN EXTERNAL INGESTION PATCH ---
# External ingestion engine: delegates document chunking, embedding, and vector
# storage to an external HTTP service instead of running save_docs_to_vector_db
# in-process.
# --- END EXTERNAL INGESTION PATCH ---

import logging
from types import SimpleNamespace

import requests
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, REQUESTS_VERIFY
from open_webui.models.config import Config
from open_webui.utils.headers import include_user_info_headers

log = logging.getLogger(__name__)


# Maps our config field names to their storage keys in the per-key Config
# store. The storage keys are unchanged from the pre-v0.11.0 ConfigVar
# declarations, so values persisted before the upgrade carry over via the
# 3ff2c63645b8 (reshape config to per-key rows) migration.
EXTERNAL_RAG_CONFIG_KEYS = {
    'RAG_RETRIEVAL_ENGINE': 'rag.retrieval_engine',
    'RAG_EXTERNAL_RETRIEVAL_URL': 'rag.external_retrieval_url',
    'RAG_EXTERNAL_RETRIEVAL_API_KEY': 'rag.external_retrieval_api_key',
    'RAG_EXTERNAL_RETRIEVAL_TIMEOUT': 'rag.external_retrieval_timeout',
    'RAG_EXTERNAL_BYPASS_QUERY_GENERATION': 'rag.external_bypass_query_generation',
    'RAG_EXTERNAL_MESSAGE_COUNT': 'rag.external_message_count',
    'RAG_EXTERNAL_USER_MESSAGES_ONLY': 'rag.external_user_messages_only',
    'EXTERNAL_INGESTION_ENGINE': 'rag.external_ingestion_engine',
    'EXTERNAL_INGESTION_URL': 'rag.external_ingestion_url',
    'EXTERNAL_INGESTION_API_KEY': 'rag.external_ingestion_api_key',
    'EXTERNAL_INGESTION_TIMEOUT': 'rag.external_ingestion_timeout',
}


async def get_external_rag_config() -> SimpleNamespace:
    """Read our external-RAG keys from the per-key Config store.

    For call sites that don't already have a RetrievalConfig in scope
    (retrieval/utils.py, tools/builtin.py, routers/files.py,
    routers/knowledge.py). Missing keys degrade to None.
    """
    values = await Config.get_many(*EXTERNAL_RAG_CONFIG_KEYS.values())
    return SimpleNamespace(**{field: values.get(key) for field, key in EXTERNAL_RAG_CONFIG_KEYS.items()})


def trim_messages_for_external(messages: list[dict], *, count: int, user_only: bool) -> list[dict]:
    """Trim the conversation before forwarding it to the external retrieval
    service, honouring RAG_EXTERNAL_MESSAGE_COUNT / RAG_EXTERNAL_USER_MESSAGES_ONLY.

    Shared by utils/middleware.py (chat_completion_files_handler) and
    tools/builtin.py (query_knowledge_files) so both paths trim identically.
    """
    candidates = [m for m in messages if m.get('role') == 'user'] if user_only else messages
    return [{'role': m.get('role', ''), 'content': m.get('content', '')} for m in candidates[-count:]]


def query_external_retrieval(
    url: str,
    api_key: str,
    queries: list[str],
    collection_names: list[str],
    k: int,
    timeout: str | None = None,
    user=None,
    messages: list[dict] | None = None,
) -> dict | None:
    """
    Query an external retrieval service.

    POST {url}/search with queries + collection_names + k.
    Optionally includes the chat messages so the external service can
    extract/generate its own queries from the conversation. Open WebUI's
    QUERY_GENERATION_PROMPT_TEMPLATE is intentionally NOT forwarded — the
    external service runs its own query generation.
    Returns dict with keys: documents, metadatas, distances (matching internal format).
    Returns None on error.
    """
    payload = {
        'queries': queries,
        'collection_names': collection_names,
        'k': k,
    }

    if messages is not None:
        payload['messages'] = messages

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        request_timeout = int(timeout) if timeout else None

        log.info(
            f'query_external_retrieval: url={url}, queries={queries}, '
            f'messages={len(messages) if messages else 0}, '
            f'collections={collection_names}, k={k}'
        )

        r = requests.post(
            f'{url}/search',
            headers=headers,
            json=payload,
            timeout=request_timeout,
            verify=REQUESTS_VERIFY,
        )
        r.raise_for_status()
        data = r.json()

        if 'documents' in data:
            return data
        else:
            log.error('No documents found in external retrieval response')
            return None

    except Exception as e:
        log.exception(f'Error in external retrieval: {e}')
        return None


# --- BEGIN EXTERNAL INGESTION PATCH ---
def process_file_external_ingestion(
    url: str,
    api_key: str,
    file_id: str,
    filename: str,
    collection_name: str,
    user_id: str,
    local_file_path: str | None = None,
    s3_bucket: str | None = None,
    s3_key: str | None = None,
    timeout: int | None = 300,
) -> dict | None:
    """
    Delegate document ingestion to an external HTTP service.

    PUT {url}/api/v1/ingest with either:
      - JSON body containing s3_bucket + s3_key (preferred when storage is S3)
      - multipart body containing the file (fallback when storage is local)

    Returns the service's response dict on success
    ({"status": True, "collection_name": ..., "chunks_count": ...})
    or None on transport / server error. Caller treats None as failure.
    """
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        endpoint = f'{url.rstrip("/")}/api/v1/ingest'

        if s3_bucket and s3_key:
            payload = {
                's3_bucket': s3_bucket,
                's3_key': s3_key,
                'file_id': file_id,
                'filename': filename,
                'collection_name': collection_name,
                'collection_type': 'file',
                'user_id': user_id,
                'overwrite': True,
            }
            log.info(
                f'process_file_external_ingestion (s3): file_id={file_id}, '
                f'collection={collection_name}, s3={s3_bucket}/{s3_key}'
            )
            r = requests.put(
                endpoint,
                headers={**headers, 'Content-Type': 'application/json'},
                json=payload,
                timeout=timeout,
                verify=REQUESTS_VERIFY,
            )
        elif local_file_path:
            log.info(
                f'process_file_external_ingestion (multipart): file_id={file_id}, '
                f'collection={collection_name}, path={local_file_path}'
            )
            with open(local_file_path, 'rb') as fh:
                files = {'file': (filename, fh)}
                data = {
                    'file_id': file_id,
                    'filename': filename,
                    'collection_name': collection_name,
                    'collection_type': 'file',
                    'user_id': user_id,
                    'overwrite': 'true',
                }
                r = requests.put(
                    endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                    verify=REQUESTS_VERIFY,
                )
        else:
            log.error('process_file_external_ingestion: no S3 reference and no local file path')
            return None

        r.raise_for_status()
        return r.json()

    except Exception as e:
        log.exception(f'Error in external ingestion: {e}')
        return None


def delete_file_external_ingestion(
    url: str,
    api_key: str,
    file_id: str,
    timeout: int = 300,
) -> dict | None:
    """Tell the external ingestion service to drop a file's vectors.

    DELETE {url}/api/v1/documents/{file_id}. The vector store lives behind the
    external service now, so Open WebUI's own vector-DB cleanup no longer reaches
    it — this call keeps the two in sync when a file is deleted.

    Best-effort: returns the service's response dict on success, or None on any
    transport/server error (logged, never raised). Callers MUST NOT let a failed
    cleanup block the user's file deletion. ``file_id`` is the bare file UUID —
    the service matches on meta.file_id, not the "file-" collection name.
    """
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        endpoint = f'{url.rstrip("/")}/api/v1/documents/{file_id}'
        log.info(f'delete_file_external_ingestion: file_id={file_id}')
        r = requests.delete(
            endpoint,
            headers=headers,
            timeout=timeout,
            verify=REQUESTS_VERIFY,
        )
        r.raise_for_status()
        return r.json()

    except Exception as e:
        log.exception(f'Error in external ingestion delete: {e}')
        return None


# --- END EXTERNAL INGESTION PATCH ---

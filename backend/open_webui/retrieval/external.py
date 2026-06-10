# --- BEGIN EXTERNAL RETRIEVAL PATCH ---
# External retrieval engine: delegates document search to an external HTTP service
# instead of querying the built-in vector DB directly.
# --- END EXTERNAL RETRIEVAL PATCH ---
# --- BEGIN EXTERNAL INGESTION PATCH ---
# External ingestion engine: delegates document chunking, embedding, and vector
# storage to an external HTTP service instead of running save_docs_to_vector_db
# in-process.
# --- END EXTERNAL INGESTION PATCH ---

import logging
from typing import Optional, List

import requests

from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, REQUESTS_VERIFY
from open_webui.utils.headers import include_user_info_headers

log = logging.getLogger(__name__)


def query_external_retrieval(
    url: str,
    api_key: str,
    queries: List[str],
    collection_names: List[str],
    k: int,
    timeout: Optional[str] = None,
    user=None,
    messages: Optional[List[dict]] = None,
) -> Optional[dict]:
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
        "queries": queries,
        "collection_names": collection_names,
        "k": k,
    }

    if messages is not None:
        payload["messages"] = messages

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        request_timeout = int(timeout) if timeout else None

        log.info(
            f"query_external_retrieval: url={url}, queries={queries}, "
            f"messages={len(messages) if messages else 0}, "
            f"collections={collection_names}, k={k}"
        )

        r = requests.post(
            f"{url}/search",
            headers=headers,
            json=payload,
            timeout=request_timeout,
            verify=REQUESTS_VERIFY,
        )
        r.raise_for_status()
        data = r.json()

        if "documents" in data:
            return data
        else:
            log.error("No documents found in external retrieval response")
            return None

    except Exception as e:
        log.exception(f"Error in external retrieval: {e}")
        return None


# --- BEGIN EXTERNAL INGESTION PATCH ---
def process_file_external_ingestion(
    url: str,
    api_key: str,
    file_id: str,
    filename: str,
    collection_name: str,
    user_id: str,
    local_file_path: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    timeout: Optional[int] = 300,
) -> Optional[dict]:
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
        headers = {"Authorization": f"Bearer {api_key}"}
        endpoint = f"{url.rstrip('/')}/api/v1/ingest"

        if s3_bucket and s3_key:
            payload = {
                "s3_bucket": s3_bucket,
                "s3_key": s3_key,
                "file_id": file_id,
                "filename": filename,
                "collection_name": collection_name,
                "collection_type": "file",
                "user_id": user_id,
                "overwrite": True,
            }
            log.info(
                f"process_file_external_ingestion (s3): file_id={file_id}, "
                f"collection={collection_name}, s3={s3_bucket}/{s3_key}"
            )
            r = requests.put(
                endpoint,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
                verify=REQUESTS_VERIFY,
            )
        elif local_file_path:
            log.info(
                f"process_file_external_ingestion (multipart): file_id={file_id}, "
                f"collection={collection_name}, path={local_file_path}"
            )
            with open(local_file_path, "rb") as fh:
                files = {"file": (filename, fh)}
                data = {
                    "file_id": file_id,
                    "filename": filename,
                    "collection_name": collection_name,
                    "collection_type": "file",
                    "user_id": user_id,
                    "overwrite": "true",
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
            log.error(
                "process_file_external_ingestion: no S3 reference and no local file path"
            )
            return None

        r.raise_for_status()
        return r.json()

    except Exception as e:
        log.exception(f"Error in external ingestion: {e}")
        return None
# --- END EXTERNAL INGESTION PATCH ---
